#!/usr/bin/env python3
"""
Test: Error recovery for GANSU-UI backend.

Verifies that after an error, subsequent calculations work without server restart.
Run with the backend already running:
    python test_error_recovery.py [base_url]
"""

import sys
import json
import time
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def _post(path, body):
    """POST JSON and return parsed response."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"    -> POST {path} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
            result = json.loads(raw)
            elapsed = time.time() - t0
            ok = result.get("ok", "?")
            print(f"ok={ok} ({elapsed:.1f}s)", flush=True)
            return result
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        body_text = e.read().decode() if e.fp else ""
        print(f"HTTP {e.code} ({elapsed:.1f}s)", flush=True)
        try:
            return json.loads(body_text)
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}: {body_text[:200]}"}


def _post_stream(path, body):
    """POST JSON and return SSE events."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"    -> POST {path} (stream) ...", end=" ", flush=True)
    t0 = time.time()
    events = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    elapsed = time.time() - t0
    types = [e["type"] for e in events[-3:]] if events else []
    print(f"{len(events)} events, last: {types} ({elapsed:.1f}s)", flush=True)
    return events

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS: {name}")
    except AssertionError as e:
        FAIL += 1
        print(f"  FAIL: {name} — {e}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL: {name} — {type(e).__name__}: {e}")


# ── Valid XYZ ──

H2O_XYZ = """3
H2O
O  0.000000  0.000000  0.117300
H  0.000000  0.756950 -0.469200
H  0.000000 -0.756950 -0.469200"""

H2_XYZ = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.74"""

# ── Invalid inputs (should cause errors) ──

BAD_XYZ = """1
bad
Zz  0.0  0.0  0.0"""

EMPTY_XYZ = ""

# Too-close atoms (may cause numerical issues)
CLOSE_XYZ = """2
close
H  0.0  0.0  0.0
H  0.0  0.0  0.01"""


# ====================================================================
# Test 1: Subprocess endpoint — /api/run
# ====================================================================

print("\n=== Test 1: /api/run — subprocess error recovery ===")

def test_run_normal():
    data = _post("/api/run", {
        "xyz_text": H2O_XYZ, "basis": "sto-3g", "method": "RHF", "timeout": 60,
    })
    assert data["ok"], f"Expected ok, got: {data.get('error', data)}"
    assert "summary" in data, "Missing summary"

def test_run_bad_xyz():
    data = _post("/api/run", {
        "xyz_text": BAD_XYZ, "basis": "sto-3g", "method": "RHF", "timeout": 30,
    })
    assert not data["ok"], "Expected failure for bad XYZ"

def test_run_bad_basis():
    data = _post("/api/run", {
        "xyz_text": H2O_XYZ, "basis": "nonexistent-basis-999", "method": "RHF", "timeout": 30,
    })
    assert not data["ok"], "Expected failure for nonexistent basis"

def test_run_after_error():
    """Key test: normal calc works after previous error."""
    data = _post("/api/run", {
        "xyz_text": H2O_XYZ, "basis": "sto-3g", "method": "RHF", "timeout": 60,
    })
    assert data["ok"], f"Expected ok after error recovery, got: {data.get('error', data)}"

test("Normal H2O calc", test_run_normal)
test("Bad XYZ → error", test_run_bad_xyz)
test("Normal calc after bad XYZ", test_run_after_error)
test("Bad basis → error", test_run_bad_basis)
test("Normal calc after bad basis", test_run_after_error)


# ====================================================================
# Test 2: Subprocess stream endpoint — /api/run/stream
# ====================================================================

print("\n=== Test 2: /api/run/stream — SSE error recovery ===")

def test_stream_normal():
    events = _post_stream("/api/run/stream", {
        "xyz_text": H2_XYZ, "basis": "sto-3g", "method": "RHF", "timeout": 60,
    })
    has_result = any(e["type"] == "result" for e in events)
    assert has_result, "No result event in stream"

def test_stream_bad():
    events = _post_stream("/api/run/stream", {
        "xyz_text": BAD_XYZ, "basis": "sto-3g", "method": "RHF", "timeout": 30,
    })
    has_error = any(e["type"] == "error" for e in events)
    assert has_error, "Expected error event in stream"

def test_stream_after_error():
    events = _post_stream("/api/run/stream", {
        "xyz_text": H2_XYZ, "basis": "sto-3g", "method": "RHF", "timeout": 60,
    })
    has_result = any(e["type"] == "result" for e in events)
    assert has_result, "No result event after error recovery"

test("Stream normal H2 calc", test_stream_normal)
test("Stream bad XYZ → error", test_stream_bad)
test("Stream normal after error", test_stream_after_error)


# ====================================================================
# Test 3: PES in-process — /api/pes/point (most critical)
# ====================================================================

print("\n=== Test 3: /api/pes/point — in-process error recovery ===")

def pes_point(xyz, basis="sto-3g", use_prev=False):
    return _post("/api/pes/point", {
        "xyz_text": xyz, "basis": basis, "method": "RHF",
        "charge": 0, "post_hf_method": "none",
        "use_prev_density": use_prev, "timeout": 60,
    })

def pes_reset():
    _post("/api/pes/reset", {})

def test_pes_normal():
    pes_reset()
    data = pes_point(H2_XYZ)
    assert data["ok"], f"Expected ok, got: {data.get('error', data)}"
    assert data["energy"] < 0, f"Energy should be negative: {data['energy']}"

def test_pes_bad_xyz():
    data = pes_point(BAD_XYZ)
    assert not data["ok"], "Expected failure for bad XYZ"

def test_pes_bad_basis():
    data = pes_point(H2_XYZ, basis="nonexistent-basis-999")
    assert not data["ok"], "Expected failure for nonexistent basis"

def test_pes_after_error():
    """Key test: PES works after previous in-process error."""
    data = pes_point(H2_XYZ)
    assert data["ok"], f"Expected ok after error, got: {data.get('error', data)}"
    assert data["energy"] < 0, f"Energy should be negative: {data['energy']}"

def test_pes_density_cleared_after_error():
    """After error, density cache should be cleared."""
    pes_reset()
    # First: normal point
    d1 = pes_point(H2_XYZ)
    assert d1["ok"], "First point should succeed"
    # Second: error
    d2 = pes_point(BAD_XYZ, use_prev=True)
    assert not d2["ok"], "Bad point should fail"
    # Third: normal again — should NOT use stale density
    d3 = pes_point(H2_XYZ, use_prev=False)
    assert d3["ok"], f"Recovery point should succeed, got: {d3.get('error')}"

def test_pes_consecutive_errors():
    """Multiple errors in a row, then recovery."""
    pes_reset()
    for i in range(3):
        d = pes_point(BAD_XYZ)
        assert not d["ok"], f"Error {i+1} should fail"
    d = pes_point(H2_XYZ)
    assert d["ok"], f"Should recover after 3 errors, got: {d.get('error')}"

def test_pes_scan_sequence():
    """Simulate a PES scan: multiple points with density reuse."""
    pes_reset()
    energies = []
    for r_val in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        xyz = f"2\nH2 R={r_val}\nH  0.0  0.0  0.0\nH  0.0  0.0  {r_val}"
        d = pes_point(xyz, use_prev=len(energies) > 0)
        assert d["ok"], f"PES point R={r_val} failed: {d.get('error')}"
        energies.append(d["energy"])
    # Energies should all be different (no stale state)
    assert len(set(f"{e:.6f}" for e in energies)) > 1, "All energies identical — stale state?"

test("PES normal H2 point", test_pes_normal)
test("PES bad XYZ → error", test_pes_bad_xyz)
test("PES normal after bad XYZ", test_pes_after_error)
test("PES bad basis → error", test_pes_bad_basis)
test("PES normal after bad basis", test_pes_after_error)
test("PES density cleared after error", test_pes_density_cleared_after_error)
test("PES consecutive errors then recovery", test_pes_consecutive_errors)
test("PES scan sequence with density reuse", test_pes_scan_sequence)


# ====================================================================
# Test 4: GPU error recovery (large basis → OOM → recovery)
# ====================================================================

print("\n=== Test 4: GPU error recovery (large basis) ===")

# Large molecule + large basis that may OOM
LARGE_XYZ = """6
C2H4
C  0.000000  0.000000  0.667000
C  0.000000  0.000000 -0.667000
H  0.000000  0.923000  1.238000
H  0.000000 -0.923000  1.238000
H  0.000000  0.923000 -1.238000
H  0.000000 -0.923000 -1.238000"""

def test_inprocess_normal():
    data = _post("/api/run/inprocess", {
        "xyz_text": H2O_XYZ, "basis": "sto-3g", "method": "RHF",
    })
    assert data["ok"], f"Expected ok, got: {data.get('error', data)}"

def test_inprocess_large_basis():
    """Try a large basis — may OOM on GPU. Should not crash server."""
    data = _post("/api/run/inprocess", {
        "xyz_text": LARGE_XYZ, "basis": "cc-pvqz", "method": "RHF",
    })
    # May succeed or fail depending on GPU memory — either is fine
    print(f"      (large basis: ok={data.get('ok')}, error={data.get('error', 'none')[:60]})")

def test_inprocess_after_large():
    """Key test: normal calc works after large basis attempt."""
    data = _post("/api/run/inprocess", {
        "xyz_text": H2O_XYZ, "basis": "sto-3g", "method": "RHF",
    })
    assert data["ok"], f"Expected ok after GPU recovery, got: {data.get('error', data)}"

def test_pes_after_inprocess_error():
    """PES should also work after in-process GPU error."""
    pes_reset()
    data = pes_point(H2_XYZ)
    assert data["ok"], f"PES should work after recovery, got: {data.get('error', data)}"

test("In-process normal calc", test_inprocess_normal)
test("In-process large basis (may OOM)", test_inprocess_large_basis)
test("In-process normal after large basis", test_inprocess_after_large)
test("PES after in-process error", test_pes_after_inprocess_error)


# ====================================================================
# Test 5: /api/pes/reset clears everything
# ====================================================================

print("\n=== Test 5: /api/pes/reset — full state clear ===")

def test_reset_then_run():
    pes_reset()
    time.sleep(0.5)
    data = pes_point(H2O_XYZ)
    assert data["ok"], f"Should work after reset, got: {data.get('error')}"

test("Reset then normal run", test_reset_then_run)


# ====================================================================
# Summary
# ====================================================================

print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED — check output above")
    sys.exit(1)
