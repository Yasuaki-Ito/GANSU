# GANSU C API Reference

## Overview

The GANSU C API (`gansu_api.h`) provides a stable, language-agnostic interface to GANSU's quantum chemistry engine. It uses opaque handles and `extern "C"` linkage, making it callable from virtually any programming language.

**Shared library**: `libgansu.so` (Linux) / `libgansu.dylib` (macOS)

**Header**: `include/gansu_api.h`

## Build

```bash
cd build
cmake ..
make gansu_shared    # produces libgansu.so
```

---

## API Reference

### Lifecycle

#### `gansu_init`
```c
void gansu_init(int force_cpu);
```
Initialize the GANSU runtime. Call once before any other function.
- `force_cpu = 0`: Auto-detect GPU
- `force_cpu = 1`: Force CPU-only mode

#### `gansu_finalize`
```c
void gansu_finalize(void);
```
Finalize the GANSU runtime. Call at program exit.

#### `gansu_create`
```c
gansu_handle_t gansu_create(void);
```
Create a new calculation context. Returns an opaque handle.

#### `gansu_destroy`
```c
void gansu_destroy(gansu_handle_t h);
```
Destroy a calculation context and free all associated resources.

---

### Configuration

#### `gansu_set`
```c
int gansu_set(gansu_handle_t h, const char* key, const char* value);
```
Set an arbitrary parameter. Returns 0 on success.

Common keys: `"xyzfilename"`, `"gbsfilename"`, `"method"`, `"post_hf_method"`, `"run_type"`, `"eri_method"`, `"initial_guess"`, `"convergence_method"`, `"n_excited_states"`, `"optimizer"`, `"quiet"`.

See [parameters.md](parameters.md) for the full list.

Special key: `"quiet"` — set to `"true"` to suppress stdout during `gansu_run`.

#### Convenience functions
```c
int gansu_set_xyz(gansu_handle_t h, const char* path);      // set xyzfilename
int gansu_set_basis(gansu_handle_t h, const char* path);     // set gbsfilename
int gansu_set_method(gansu_handle_t h, const char* method);  // set method
int gansu_set_post_hf(gansu_handle_t h, const char* post_hf); // set post_hf_method
```

---

### Execution

#### `gansu_run`
```c
int gansu_run(gansu_handle_t h);
```
Run the calculation (SCF + post-HF if configured). Returns 0 on success, nonzero on error.

---

### Results

#### `gansu_get_total_energy`
```c
double gansu_get_total_energy(gansu_handle_t h);
```
HF total energy (electronic + nuclear repulsion) in Hartree.

#### `gansu_get_post_hf_energy`
```c
double gansu_get_post_hf_energy(gansu_handle_t h);
```
Post-HF correlation energy in Hartree. Returns 0 if no post-HF method was used.

#### `gansu_get_nuclear_repulsion_energy`
```c
double gansu_get_nuclear_repulsion_energy(gansu_handle_t h);
```
Nuclear repulsion energy in Hartree.

#### `gansu_get_num_basis`
```c
int gansu_get_num_basis(gansu_handle_t h);
```
Number of basis functions (nao).

#### `gansu_get_num_electrons`
```c
int gansu_get_num_electrons(gansu_handle_t h);
```
Number of electrons.

#### `gansu_get_num_atoms`
```c
int gansu_get_num_atoms(gansu_handle_t h);
```
Number of atoms.

#### `gansu_get_orbital_energies`
```c
int gansu_get_orbital_energies(gansu_handle_t h, double* buf, int buf_size);
```
Copy orbital energies into `buf`. Returns number of values written, or -1 on error. Buffer must have at least `num_basis` elements.

#### `gansu_get_mo_coefficients`
```c
int gansu_get_mo_coefficients(gansu_handle_t h, double* buf, int buf_size);
```
Copy MO coefficient matrix (nao x nao, row-major) into `buf`. Returns `nao*nao` on success.

#### `gansu_get_ccsd_1rdm_mo`
```c
int gansu_get_ccsd_1rdm_mo(gansu_handle_t h, double* buf, int buf_size);
```
Copy CCSD 1-RDM in MO basis (nao x nao, row-major) into `buf`. Only available after running with `post_hf_method = "ccsd_density"`.

#### `gansu_get_excited_state_report`
```c
const char* gansu_get_excited_state_report(gansu_handle_t h);
```
Returns a formatted string with excited state energies, oscillator strengths, and dominant transitions. Pointer is valid until `gansu_destroy`.

---

## Usage Examples

### C

```c
#include "gansu_api.h"
#include <stdio.h>

int main() {
    gansu_init(0);

    gansu_handle_t h = gansu_create();
    gansu_set_xyz(h, "H2O.xyz");
    gansu_set_basis(h, "cc-pvdz");
    gansu_set_method(h, "RHF");
    gansu_set_post_hf(h, "ccsd");
    gansu_set(h, "quiet", "true");

    if (gansu_run(h) == 0) {
        double e_hf   = gansu_get_total_energy(h);
        double e_corr = gansu_get_post_hf_energy(h);
        printf("HF energy:   %.8f Hartree\n", e_hf);
        printf("CCSD corr:   %.8f Hartree\n", e_corr);
        printf("Total:       %.8f Hartree\n", e_hf + e_corr);
        printf("nao=%d, ne=%d\n", gansu_get_num_basis(h), gansu_get_num_electrons(h));
    }

    gansu_destroy(h);
    gansu_finalize();
    return 0;
}
```

Compile:
```bash
gcc -o my_calc my_calc.c -L/path/to/build -lgansu -lstdc++ -lm
```

### Rust

```rust
use std::ffi::CString;
use std::os::raw::{c_int, c_double, c_char, c_void};

extern "C" {
    fn gansu_init(force_cpu: c_int);
    fn gansu_finalize();
    fn gansu_create() -> *mut c_void;
    fn gansu_destroy(h: *mut c_void);
    fn gansu_set_xyz(h: *mut c_void, path: *const c_char) -> c_int;
    fn gansu_set_basis(h: *mut c_void, path: *const c_char) -> c_int;
    fn gansu_set(h: *mut c_void, key: *const c_char, val: *const c_char) -> c_int;
    fn gansu_run(h: *mut c_void) -> c_int;
    fn gansu_get_total_energy(h: *mut c_void) -> c_double;
    fn gansu_get_post_hf_energy(h: *mut c_void) -> c_double;
}

fn main() {
    unsafe {
        gansu_init(0);
        let h = gansu_create();
        let xyz = CString::new("H2O.xyz").unwrap();
        let basis = CString::new("cc-pvdz").unwrap();
        let quiet_k = CString::new("quiet").unwrap();
        let quiet_v = CString::new("true").unwrap();
        let post = CString::new("post_hf_method").unwrap();
        let ccsd = CString::new("ccsd").unwrap();

        gansu_set_xyz(h, xyz.as_ptr());
        gansu_set_basis(h, basis.as_ptr());
        gansu_set(h, quiet_k.as_ptr(), quiet_v.as_ptr());
        gansu_set(h, post.as_ptr(), ccsd.as_ptr());
        gansu_run(h);

        let e = gansu_get_total_energy(h) + gansu_get_post_hf_energy(h);
        println!("E = {:.8} Hartree", e);

        gansu_destroy(h);
        gansu_finalize();
    }
}
```

### Julia

```julia
const lib = "libgansu.so"

ccall((:gansu_init, lib), Cvoid, (Cint,), 0)
h = ccall((:gansu_create, lib), Ptr{Cvoid}, ())

ccall((:gansu_set_xyz, lib), Cint, (Ptr{Cvoid}, Cstring), h, "H2O.xyz")
ccall((:gansu_set_basis, lib), Cint, (Ptr{Cvoid}, Cstring), h, "cc-pvdz")
ccall((:gansu_set, lib), Cint, (Ptr{Cvoid}, Cstring, Cstring), h, "post_hf_method", "ccsd")
ccall((:gansu_set, lib), Cint, (Ptr{Cvoid}, Cstring, Cstring), h, "quiet", "true")
ccall((:gansu_run, lib), Cint, (Ptr{Cvoid},), h)

e_hf   = ccall((:gansu_get_total_energy, lib), Cdouble, (Ptr{Cvoid},), h)
e_corr = ccall((:gansu_get_post_hf_energy, lib), Cdouble, (Ptr{Cvoid},), h)
println("E = $(e_hf + e_corr) Hartree")

ccall((:gansu_destroy, lib), Cvoid, (Ptr{Cvoid},), h)
ccall((:gansu_finalize, lib), Cvoid, ())
```

### JavaScript (Node.js with ffi-napi)

```javascript
const ffi = require('ffi-napi');
const ref = require('ref-napi');

const gansu = ffi.Library('./libgansu.so', {
    'gansu_init':               ['void',   ['int']],
    'gansu_finalize':           ['void',   []],
    'gansu_create':             ['pointer', []],
    'gansu_destroy':            ['void',   ['pointer']],
    'gansu_set_xyz':            ['int',    ['pointer', 'string']],
    'gansu_set_basis':          ['int',    ['pointer', 'string']],
    'gansu_set':                ['int',    ['pointer', 'string', 'string']],
    'gansu_run':                ['int',    ['pointer']],
    'gansu_get_total_energy':   ['double', ['pointer']],
    'gansu_get_post_hf_energy': ['double', ['pointer']],
});

gansu.gansu_init(0);
const h = gansu.gansu_create();
gansu.gansu_set_xyz(h, 'H2O.xyz');
gansu.gansu_set_basis(h, 'cc-pvdz');
gansu.gansu_set(h, 'post_hf_method', 'ccsd');
gansu.gansu_set(h, 'quiet', 'true');
gansu.gansu_run(h);

const E = gansu.gansu_get_total_energy(h) + gansu.gansu_get_post_hf_energy(h);
console.log(`E = ${E.toFixed(8)} Hartree`);

gansu.gansu_destroy(h);
gansu.gansu_finalize();
```

---

## Thread Safety

The C API is **NOT thread-safe**. GPU state is global. Do not call `gansu_run` concurrently from multiple threads. Sequential calls with separate handles are safe.

## Error Handling

- Functions returning `int` return 0 on success, nonzero on error.
- Functions returning `double` return 0.0 if the handle is invalid or calculation has not been run.
- Error messages are printed to stderr.
