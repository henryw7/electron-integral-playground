
def binomial_coefficient(n: int, m: int) -> int:
    assert 0 <= m and m <= n

    if m == 0 or m == n:
        return 1

    return binomial_coefficient(n - 1, m - 1) + binomial_coefficient(n - 1, m)


if __name__ == "__main__":
    for n in range(26 + 1):
        for m in range(n + 1):
            print(f"{binomial_coefficient(n, m):8d}, ", end = "")
        print()
