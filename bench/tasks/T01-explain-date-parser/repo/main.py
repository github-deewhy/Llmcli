import sys

from report import count_per_day


def main() -> None:
    for day, count in sorted(count_per_day(sys.stdin.readlines()).items()):
        print(day, count)


if __name__ == "__main__":
    main()
