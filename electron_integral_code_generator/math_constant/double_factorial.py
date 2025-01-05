
def double_factorial(n: int) -> int:
    assert n > -1

    if n == 0 or n == 1:
        return 1

    return n * double_factorial(n - 2)


if __name__ == "__main__":
    for n in range(24 + 2 + 1):
        print(f"{double_factorial(n):14d}, ", end = "")
        if (n % 10 == 0):
            print()
    print()