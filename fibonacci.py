def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def main():
    n = 10  # Calculate first 10 Fibonacci numbers
    print(f"First {n} Fibonacci numbers:")
    fib_numbers = [fibonacci(i) for i in range(n)]
    print(", ".join(map(str, fib_numbers)))

if __name__ == "__main__":
    main()
