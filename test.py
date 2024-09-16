import itertools

if __name__ == "__main__":
    import itertools

    t = ["a", "b", "c"]

    combinations = []
    for level in range(2, 3 + 1):
        for comb in itertools.combinations(t, level):
            combinations.append(comb)

    print(combinations)