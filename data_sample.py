import random

def generate_random_numbers_to_file(filename, columns, lines, min_value, max_value):
    try:
        with open(filename, 'w') as file:
            for _ in range(lines):
                line = [random.uniform(min_value, max_value) for _ in range(columns)]
                file.write(' '.join(str(value) for value in line) + ' {0} {1}\n'.format(min_value, max_value))

    except IOError as e:
        print(f"Error writing to file: {e}")

def uniform(min, max, n):
    return [random.uniform(min, max) for _ in range(n)]

if __name__ == "__main__":
    output_file = "data.txt"  # Name of the output file
    columns = 10  # Number of random numbers to generate per line
    lines = 200
    minimum_value = 0                   # Minimum possible random number
    maximum_value = 1               # Maximum possible random number

    generate_random_numbers_to_file(output_file, columns, lines, minimum_value, maximum_value)