# FSGe CI Test Reference Data (nmax=2)

This directory contains reference data for the GitHub Actions CI/CD testing workflow.

## Overview

The reference data is used to validate that code changes don't break the FSGe simulation convergence behavior. The test uses a reduced configuration (`nmax=2`) for faster execution.

## Generating Reference Data

To generate new reference data, follow these steps:

### 1. Set up Docker environment

```bash
cd /path/to/svFSGe
mkdir -p svMultiPhysics

docker run -it --user $(id -u):$(id -g) \
           -v ./svMultiPhysics:/svfsi \
           -v .:/svFSGe \
           simvascular/solver:latest /bin/bash
```

### 2. Build svFSIplus (inside container)

```bash
cd /svfsi
git init
git remote add origin https://github.com/Eleven7825/svMultiPhysics.git
git fetch origin FSGe
git checkout -b FSGe origin/FSGe
bash makeCommand.sh
```

### 3. Run the test configuration (inside container)

```bash
cd /svFSGe
python3 ./fsg.py in_sim/partitioned_test.json
```

This will take approximately 20-60 minutes depending on your hardware.

### 4. Extract reference data (on host)

After the test completes, a new directory `partitioned_<timestamp>/` will be created. Extract the convergence data:

```bash
# Find the most recent partitioned directory
RESULT_DIR=$(ls -dt partitioned_* | head -1)

# Extract just the error/convergence data
python3 << 'EOF'
import json

# Load the full results
with open(f"${RESULT_DIR}/partitioned.json", 'r') as f:
    full_data = json.load(f)

# Create reference data structure
reference_data = {
    "description": "Reference convergence data for FSGe CI test with nmax=2",
    "configuration": {
        "nmax": full_data.get("nmax", 2),
        "n_procs": full_data.get("n_procs", {}),
        "coup": {
            "method": full_data.get("coup", {}).get("method", ""),
            "tol": full_data.get("coup", {}).get("tol", 0)
        }
    },
    "convergence": {
        "error": full_data.get("error", {})
    }
}

# Save to reference file
with open("test_reference/nmax_2/reference_results.json", 'w') as f:
    json.dump(reference_data, f, indent=2)

print("Reference data saved to test_reference/nmax_2/reference_results.json")
EOF
```

### 5. Verify the reference data

```bash
# Test the comparison script
python3 scripts/compare_results.py \
    test_reference/nmax_2/reference_results.json \
    ${RESULT_DIR}/partitioned.json
```

This should show all comparisons passing.

### 6. Commit the reference data

```bash
git add test_reference/nmax_2/reference_results.json
git commit -m "Update FSGe test reference data"
```

## Reference Data Format

The `reference_results.json` file contains:

```json
{
  "description": "Reference convergence data for FSGe CI test with nmax=2",
  "configuration": {
    "nmax": 2,
    "n_procs": {
      "fluid": 2,
      "mesh": 1,
      "solid": 1
    },
    "coup": {
      "method": "iqn_ils",
      "tol": 0.001
    }
  },
  "convergence": {
    "error": {
      "solid": [
        [1.0, 0.05, 0.001],  // Time step 0: iteration errors
        [0.8, 0.03, 0.0008], // Time step 1: iteration errors
        [0.7, 0.02, 0.0006]  // Time step 2: iteration errors
      ]
      // ... other fields (fluid, mesh, etc.)
    }
  }
}
```

## Comparison Tolerances

The CI test uses these tolerances when comparing against reference data:

- **Iteration count**: ±2 iterations per time step
- **Error norms**: 10% relative difference

These tolerances account for minor numerical variations across different hardware and environments.

## When to Update Reference Data

Update the reference data when:

1. **Intentional algorithm changes**: You've modified the coupling algorithm or solver parameters
2. **Expected convergence improvements**: Code changes that should improve convergence behavior
3. **Test configuration changes**: Modifications to mesh, time stepping, or other parameters
4. **Major dependency updates**: Updating svFSIplus or other core dependencies

Do NOT update reference data to "fix" a failing test without understanding why the test is failing. A failing test usually indicates a regression that should be investigated.

## Troubleshooting

### Test fails with iteration count differences

- Check if your changes affected the coupling algorithm
- Review convergence behavior in the uploaded artifacts (convergence.png)
- If the changes are expected and beneficial, generate new reference data

### Test fails with error norm differences

- Verify that the simulation completed successfully
- Check for numerical stability issues
- Review the coupling residuals

### Cannot generate reference data locally

- Ensure Docker is properly installed and running
- Verify you have sufficient disk space (~5-10 GB)
- Check that all required input files exist in `in_geo/`, `in_petsc/`, and `in_svfsi_plus/`
