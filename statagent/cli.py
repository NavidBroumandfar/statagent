"""
Command-line interface for StatAgent.

This module provides an interactive CLI for performing statistical analyses
without writing code.
"""

import sys
import argparse
import numpy as np
from typing import Optional

from statagent import (
    NegativeBinomialAnalyzer,
    SurvivalMixtureModel,
    MethodOfMoments,
    BayesianEstimator,
    ZTest,
    PolynomialRegression
)


def negative_binomial_cli(args):
    """Run Negative Binomial analysis from CLI."""
    print("\n" + "="*60)
    print("Negative Binomial Analysis")
    print("="*60)
    
    nb = NegativeBinomialAnalyzer(k=args.k, p=args.p)
    
    if args.stats:
        stats = nb.compute_statistics()
        print(f"\nStatistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Variance: {stats['variance']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Median: {stats['median']}")
    
    if args.range:
        first, last = nb.find_significant_range(threshold=args.threshold)
        print(f"\nSignificant Range (P(X=x) >= {args.threshold}):")
        print(f"  [{first}, {last}]")
    
    if args.plot:
        nb.plot_pmf(save_path=args.output)
        print(f"\nPlot saved to {args.output}")
    
    if args.summary:
        print(nb.summary())


def survival_mixture_cli(args):
    """Run Survival Mixture analysis from CLI."""
    print("\n" + "="*60)
    print("Survival Mixture Model")
    print("="*60)
    
    model = SurvivalMixtureModel(
        w1=args.w1, rate1=args.rate1,
        w2=args.w2, rate2=args.rate2
    )
    
    if args.stats:
        stats = model.compute_statistics()
        print(f"\nStatistics:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Variance: {stats['variance']:.6f}")
    
    if args.probability:
        prob = model.compute_probability(args.prob_lower, args.prob_upper)
        print(f"\nP({args.prob_lower} <= Y <= {args.prob_upper}) = {prob:.6f}")
    
    if args.simulate:
        samples = model.simulate(n=args.n_samples)
        print(f"\nSimulation (n={args.n_samples}):")
        print(f"  Mean: {np.mean(samples):.6f}")
        print(f"  Median: {np.median(samples):.6f}")
    
    if args.plot:
        model.plot_pdf(save_path=args.output)
        print(f"\nPlot saved to {args.output}")
    
    if args.summary:
        print(model.summary())


def hypothesis_test_cli(args):
    """Run hypothesis test from CLI."""
    print("\n" + "="*60)
    print("Z-Test for Population Mean")
    print("="*60)
    
    # Read data
    if args.data_file:
        data = np.loadtxt(args.data_file)
    else:
        data = np.array(args.data)
    
    test = ZTest(data, mu_0=args.mu_0, sigma=args.sigma)
    
    if args.test_type == 'left':
        result = test.left_tailed_test(alpha=args.alpha)
    elif args.test_type == 'right':
        result = test.right_tailed_test(alpha=args.alpha)
    else:
        result = test.two_tailed_test(alpha=args.alpha)
    
    print(f"\nTest Results:")
    print(f"  Sample mean: {result['sample_mean']:.4f}")
    print(f"  Z-statistic: {result['z_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Decision: {'Reject H0' if result['reject_null'] else 'Fail to reject H0'}")
    
    if args.summary:
        print(test.summary(test_type=args.test_type, alpha=args.alpha))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="StatAgent - Statistical Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis type')
    
    # Negative Binomial
    nb_parser = subparsers.add_parser('negbin', help='Negative Binomial analysis')
    nb_parser.add_argument('-k', type=int, required=True, help='Number of successes')
    nb_parser.add_argument('-p', type=float, required=True, help='Success probability')
    nb_parser.add_argument('--stats', action='store_true', help='Show statistics')
    nb_parser.add_argument('--range', action='store_true', help='Show significant range')
    nb_parser.add_argument('--threshold', type=float, default=0.005, help='Threshold for range')
    nb_parser.add_argument('--plot', action='store_true', help='Generate plot')
    nb_parser.add_argument('--output', default='output.png', help='Output file')
    nb_parser.add_argument('--summary', action='store_true', help='Show full summary')
    
    # Survival Mixture
    sm_parser = subparsers.add_parser('survival', help='Survival mixture model')
    sm_parser.add_argument('--w1', type=float, required=True, help='Weight 1')
    sm_parser.add_argument('--rate1', type=float, required=True, help='Rate 1')
    sm_parser.add_argument('--w2', type=float, required=True, help='Weight 2')
    sm_parser.add_argument('--rate2', type=float, required=True, help='Rate 2')
    sm_parser.add_argument('--stats', action='store_true', help='Show statistics')
    sm_parser.add_argument('--probability', action='store_true', help='Compute probability')
    sm_parser.add_argument('--prob-lower', type=float, default=0, help='Lower bound')
    sm_parser.add_argument('--prob-upper', type=float, default=1, help='Upper bound')
    sm_parser.add_argument('--simulate', action='store_true', help='Run simulation')
    sm_parser.add_argument('--n-samples', type=int, default=100000, help='Number of samples')
    sm_parser.add_argument('--plot', action='store_true', help='Generate plot')
    sm_parser.add_argument('--output', default='output.png', help='Output file')
    sm_parser.add_argument('--summary', action='store_true', help='Show full summary')
    
    # Hypothesis Test
    ht_parser = subparsers.add_parser('ztest', help='Z-test for mean')
    ht_parser.add_argument('--data', nargs='+', type=float, help='Sample data')
    ht_parser.add_argument('--data-file', type=str, help='File with sample data')
    ht_parser.add_argument('--mu-0', type=float, required=True, help='Null hypothesis mean')
    ht_parser.add_argument('--sigma', type=float, required=True, help='Population std dev')
    ht_parser.add_argument('--test-type', choices=['left', 'right', 'two'], 
                          default='two', help='Type of test')
    ht_parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    ht_parser.add_argument('--summary', action='store_true', help='Show full summary')
    
    args = parser.parse_args()
    
    if args.command == 'negbin':
        negative_binomial_cli(args)
    elif args.command == 'survival':
        survival_mixture_cli(args)
    elif args.command == 'ztest':
        hypothesis_test_cli(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

