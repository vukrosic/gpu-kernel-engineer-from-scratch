from gputriton.bench import run_reference_benchmarks


def main():
    results = run_reference_benchmarks()
    for name, seconds in results.items():
        print(f"{name}: {seconds:.6f}s")


if __name__ == "__main__":
    main()
