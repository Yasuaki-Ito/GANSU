"""Convert Gaussian basis set format (.gbs) to NWChem format for PySCF."""
import re, sys

def convert(infile, outfile):
    with open(infile) as f:
        lines = f.readlines()

    out = []
    current_element = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip comments and blank lines
        if not line or line.startswith('!'):
            i += 1
            continue
        # Check for element line: "H     0" or "O     0"
        m = re.match(r'^([A-Z][a-z]?)\s+0\s*$', line)
        if m:
            current_element = m.group(1)
            out.append(f'BASIS "ao basis" PRINT\n')  # header (will be deduped)
            i += 1
            continue
        # Check for shell line: "S    1   1.00"
        m = re.match(r'^([SPDFGHI])\s+(\d+)\s+([\d.]+)', line)
        if m and current_element:
            shell_type = m.group(1)
            nprim = int(m.group(2))
            out.append(f'{current_element}    {shell_type}\n')
            for j in range(nprim):
                i += 1
                vals = lines[i].strip().split()
                exp_val = vals[0]
                coeff = vals[1]
                out.append(f'      {exp_val:>20s} {coeff:>20s}\n')
            i += 1
            continue
        # End marker: "****"
        if line.startswith('****'):
            i += 1
            continue
        i += 1

    with open(outfile, 'w') as f:
        # Write NWChem format
        f.write('BASIS "ao basis" PRINT\n')
        for line in out:
            if not line.startswith('BASIS'):
                f.write(line)
        f.write('END\n')

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
    print(f"Converted {sys.argv[1]} → {sys.argv[2]}")
