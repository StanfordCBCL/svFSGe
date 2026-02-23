#!/usr/bin/env python3
"""
Comparison script for FSGe test results.

Compares convergence metrics between reference and test runs.
Used in GitHub Actions CI/CD workflow.

Usage:
    python3 compare_results.py <reference_json> <test_json>

Exit codes:
    0: Test passed (results match within tolerance)
    1: Test failed (results differ beyond tolerance or error occurred)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np


class ComparisonError(Exception):
    """Custom exception for comparison failures."""
    pass


def load_json(filepath):
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ComparisonError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ComparisonError(f"Invalid JSON in {filepath}: {e}")


def extract_convergence_data(data):
    """
    Extract convergence data from partitioned.json or reference JSON.

    Returns:
        dict: {
            'time_steps': int,
            'iterations': list of iteration counts per time step,
            'errors': dict of final error norms per field per time step
        }
    """
    try:
        # Check if this is a full partitioned.json or a reference file
        if 'error' in data:
            errors = data['error']
        elif 'convergence' in data:
            # Reference format
            errors = data['convergence']['error']
        else:
            raise ComparisonError("No 'error' or 'convergence' field found in JSON")

        # Extract iteration counts and error norms
        convergence_data = {
            'time_steps': 0,
            'iterations': [],
            'errors': {}
        }

        # Process each field (solid, fluid, etc.)
        for field_name, time_step_errors in errors.items():
            if not isinstance(time_step_errors, list):
                continue

            num_time_steps = len(time_step_errors)
            if convergence_data['time_steps'] == 0:
                convergence_data['time_steps'] = num_time_steps
            elif convergence_data['time_steps'] != num_time_steps:
                raise ComparisonError(
                    f"Inconsistent time step counts across fields: "
                    f"{convergence_data['time_steps']} vs {num_time_steps}"
                )

            convergence_data['errors'][field_name] = []

            for t, iteration_errors in enumerate(time_step_errors):
                if not isinstance(iteration_errors, list) or len(iteration_errors) == 0:
                    raise ComparisonError(
                        f"Invalid error data for {field_name} at time step {t}"
                    )

                # Store iteration count for this time step (only once)
                if len(convergence_data['iterations']) <= t:
                    convergence_data['iterations'].append(len(iteration_errors))

                # Store final error norm for this time step
                final_error = iteration_errors[-1]
                convergence_data['errors'][field_name].append(final_error)

        return convergence_data

    except KeyError as e:
        raise ComparisonError(f"Missing expected field in JSON: {e}")


def compare_time_steps(ref_data, test_data):
    """Compare number of time steps."""
    ref_steps = ref_data['time_steps']
    test_steps = test_data['time_steps']

    if ref_steps != test_steps:
        raise ComparisonError(
            f"Time step count mismatch: reference has {ref_steps}, "
            f"test has {test_steps}"
        )

    # For nmax=2, we expect 3 time steps (0, 1, 2)
    expected_steps = 3
    if ref_steps != expected_steps:
        print(f"WARNING: Expected {expected_steps} time steps, got {ref_steps}")

    return True


def compare_iterations(ref_data, test_data, tolerance=2):
    """
    Compare iteration counts per time step.

    Args:
        tolerance: Maximum allowed difference in iteration count
    """
    failures = []

    for t in range(ref_data['time_steps']):
        ref_iters = ref_data['iterations'][t]
        test_iters = test_data['iterations'][t]
        diff = abs(test_iters - ref_iters)

        if diff > tolerance:
            failures.append(
                f"  Time step {t}: reference={ref_iters} iters, "
                f"test={test_iters} iters, diff={diff} (tolerance: {tolerance})"
            )

    if failures:
        raise ComparisonError(
            "Iteration count differences exceed tolerance:\n" + "\n".join(failures)
        )

    return True


def compare_error_norms(ref_data, test_data, rel_tolerance=0.10):
    """
    Compare final error norms per time step.

    Args:
        rel_tolerance: Relative tolerance (0.10 = 10%)
    """
    failures = []

    # Check that both have the same fields
    ref_fields = set(ref_data['errors'].keys())
    test_fields = set(test_data['errors'].keys())

    if ref_fields != test_fields:
        print(f"WARNING: Field mismatch - reference: {ref_fields}, test: {test_fields}")
        # Use intersection for comparison
        fields = ref_fields & test_fields
    else:
        fields = ref_fields

    for field in fields:
        ref_errors = ref_data['errors'][field]
        test_errors = test_data['errors'][field]

        for t in range(ref_data['time_steps']):
            ref_err = ref_errors[t]
            test_err = test_errors[t]

            # Compute relative difference
            if ref_err > 0:
                rel_diff = abs(test_err - ref_err) / ref_err
            else:
                # If reference is zero, use absolute difference
                rel_diff = abs(test_err - ref_err)

            if rel_diff > rel_tolerance:
                failures.append(
                    f"  {field} at time step {t}: reference={ref_err:.6e}, "
                    f"test={test_err:.6e}, rel_diff={rel_diff:.2%} "
                    f"(tolerance: {rel_tolerance:.1%})"
                )

    if failures:
        raise ComparisonError(
            "Error norm differences exceed tolerance:\n" + "\n".join(failures)
        )

    return True


def load_vtu(filepath):
    """Load a VTU file and return point-data as a dict of numpy arrays."""
    import meshio
    mesh = meshio.read(str(filepath))
    return mesh.point_data


# Relative tolerances from svMultiPhysics conftest.py
# https://github.com/SimVascular/svMultiPhysics/blob/main/tests/conftest.py
RTOL = {
    "Displacement": 1.0e-8,   # relaxed from conftest 1e-10 for cross-machine reproducibility
    "Velocity":     1.0e-6,   # relaxed from conftest 1e-7  for cross-machine reproducibility
}


def compare_vtu(ref_path, test_path):
    """
    Compare physical fields in two VTU files using the element-wise
    criterion from svMultiPhysics (conftest.py):

        |test - ref| <= rtol + rtol * |ref|

    i.e. rtol is used as both atol and rtol (same as np.isclose with
    atol=rtol).  Raises ComparisonError if any field has points outside
    the tolerance.
    """
    print(f"  Loading reference VTU: {ref_path}")
    ref_data = load_vtu(ref_path)
    print(f"  Loading test VTU:      {test_path}")
    test_data = load_vtu(test_path)

    print()
    print("  Field-by-field comparison:")

    msg = ""
    for field, rtol in RTOL.items():
        if field not in ref_data:
            print(f"    ? {field:20s} not in reference – skipped")
            continue
        if field not in test_data:
            raise ComparisonError(f"Field '{field}' missing in test VTU")

        a = test_data[field].flatten().astype(float)
        b = ref_data[field].flatten().astype(float)

        # Element-wise criterion (mirrors conftest.py exactly):
        #   rel_diff = |a - b| - rtol - rtol * |b|
        # A point passes when rel_diff <= 0.
        rel_diff = np.abs(a - b) - rtol - rtol * np.abs(b)

        close = rel_diff <= 0.0
        if np.all(close):
            print(f"    ✓ {field:20s} all points within rtol={rtol:.0e}")
        else:
            wrong   = 1.0 - np.sum(close) / close.size
            i_max   = rel_diff.argmax()
            max_rel = rel_diff[i_max]
            max_abs = np.abs(a[i_max] - b[i_max])

            print(f"    ✗ {field:20s} {wrong:.1%} of points exceed rtol={rtol:.0e}  "
                  f"(max rel_diff={max_rel:.2e}, max abs={max_abs:.2e})")

            msg += (f"  {field}: {wrong:.1%} of points exceed rtol={rtol:.0e}. "
                    f"Max rel_diff={max_rel:.2e}, max abs diff={max_abs:.2e}\n")

    if msg:
        raise ComparisonError(
            "VTU field differences exceed tolerance:\n" + msg.rstrip()
        )


def print_summary(ref_data, test_data):
    """Print comparison summary."""
    print("=" * 70)
    print("FSGe Test Results Comparison Summary")
    print("=" * 70)
    print()

    print(f"Time steps: {test_data['time_steps']}")
    print()

    print("Iteration counts per time step:")
    for t in range(test_data['time_steps']):
        ref_iters = ref_data['iterations'][t]
        test_iters = test_data['iterations'][t]
        diff = test_iters - ref_iters
        status = "✓" if abs(diff) <= 2 else "✗"
        print(f"  {status} Time step {t}: test={test_iters}, ref={ref_iters}, diff={diff:+d}")
    print()

    print("Final error norms per time step:")
    for field in sorted(test_data['errors'].keys()):
        print(f"  {field}:")
        ref_errors = ref_data['errors'].get(field, [])
        test_errors = test_data['errors'][field]

        for t in range(test_data['time_steps']):
            if t < len(ref_errors):
                ref_err = ref_errors[t]
                test_err = test_errors[t]
                if ref_err > 0:
                    rel_diff = abs(test_err - ref_err) / ref_err
                else:
                    rel_diff = abs(test_err - ref_err)
                status = "✓" if rel_diff <= 0.10 else "✗"
                print(f"    {status} Time step {t}: test={test_err:.6e}, "
                      f"ref={ref_err:.6e}, rel_diff={rel_diff:.2%}")
            else:
                print(f"    ? Time step {t}: test={test_errors[t]:.6e}, ref=N/A")
    print()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare FSGe test results against reference data"
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Path to reference results JSON file"
    )
    parser.add_argument(
        "test",
        type=Path,
        help="Path to test results JSON file (partitioned.json)"
    )
    parser.add_argument(
        "--iter-tolerance",
        type=int,
        default=2,
        help="Iteration count tolerance (default: 2)"
    )
    parser.add_argument(
        "--error-tolerance",
        type=float,
        default=0.10,
        help="Error norm relative tolerance (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--ref-vtu",
        type=Path,
        default=None,
        help="Path to reference VTU file (enables field comparison)"
    )
    parser.add_argument(
        "--test-vtu",
        type=Path,
        default=None,
        help="Path to test VTU file (required when --ref-vtu is given)"
    )

    args = parser.parse_args()

    try:
        # Load JSON files
        print(f"Loading reference: {args.reference}")
        ref_json = load_json(args.reference)

        print(f"Loading test results: {args.test}")
        test_json = load_json(args.test)
        print()

        # Extract convergence data
        ref_data = extract_convergence_data(ref_json)
        test_data = extract_convergence_data(test_json)

        # Print summary
        print_summary(ref_data, test_data)

        # Perform comparisons
        print("Running comparisons...")
        print()

        print("1. Comparing time step counts...")
        compare_time_steps(ref_data, test_data)
        print("   ✓ PASS: Time step counts match")
        print()

        print(f"2. Comparing iteration counts (tolerance: ±{args.iter_tolerance})...")
        compare_iterations(ref_data, test_data, args.iter_tolerance)
        print("   ✓ PASS: Iteration counts within tolerance")
        print()

        print(f"3. Comparing error norms (tolerance: {args.error_tolerance:.1%})...")
        compare_error_norms(ref_data, test_data, args.error_tolerance)
        print("   ✓ PASS: Error norms within tolerance")
        print()

        if args.ref_vtu or args.test_vtu:
            if not (args.ref_vtu and args.test_vtu):
                raise ComparisonError(
                    "Both --ref-vtu and --test-vtu must be provided together"
                )
            print("4. Comparing VTU fields...")
            compare_vtu(args.ref_vtu, args.test_vtu)
            print()
            print("   ✓ PASS: VTU fields within tolerance")
            print()

        print("=" * 70)
        print("OVERALL RESULT: PASS ✓")
        print("=" * 70)

        return 0

    except ComparisonError as e:
        print()
        print("=" * 70)
        print("COMPARISON FAILED ✗")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        return 1

    except Exception as e:
        print()
        print("=" * 70)
        print("UNEXPECTED ERROR ✗")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
