def sample_mean(a, b, c):
    try:
        a = int(a)
        b = int(b)
        c = int(c)
        mean_numbers = [a, b, c]
        d = len(mean_numbers)
        result_mean = (a + b + c)/d
        return float(result_mean)
    except ZeroDivisionError:
        print("Error: Number Not Valid")
    except ValueError:
        print("Error: Only Numeric Values")
