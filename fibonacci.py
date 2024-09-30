def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def main():
    n = 10  # Calculate first 10 Fibonacci numbers
    print(f"First {n} Fibonacci numbers:")
    for i in range(n):
        print(fibonacci(i), end=" ")

if __name__ == "__main__":
    main()
