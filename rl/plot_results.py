"""Plot an evaluation CSV without assuming a fixed map or number of humans."""
import argparse
from utils.statistics import plot_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', default='logs/eval_combined_metrics.csv')
    parser.add_argument('--output', default='logs/metrics_plot.png')
    args = parser.parse_args()
    plot_metrics(args.csv, args.output)


if __name__ == '__main__':
    main()
