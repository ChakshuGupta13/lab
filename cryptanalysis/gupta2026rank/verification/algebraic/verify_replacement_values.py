import json


DIMENSION = 256
FAULT_INDICES = (1, 2, 3)
GENERIC_REPLACEMENT = 1
EXPECTED_GENERIC_RANK = 192
EXPECTED_EXCEPTIONAL_RANK = 128
EXPECTED_ROWSPACE_RANK = 192
CONFIGURATIONS = (
    {
        "shape": "incomplete",
        "scheme": "ML-KEM",
        "modulus": 3329,
        "root": 17,
        "reversal_bits": 7,
        "minimum_length": 2,
        "exceptional_replacement": 1049,
        "first_three_twiddles": (1729, 2580, 3289),
    },
    {
        "shape": "complete",
        "scheme": "ML-DSA",
        "modulus": 8380417,
        "root": 1753,
        "reversal_bits": 8,
        "minimum_length": 1,
        "exceptional_replacement": 1600002,
        "first_three_twiddles": (4808194, 3765607, 3761513),
    },
)


def bit_reverse(value, width):
    result = 0
    for _ in range(width):
        result = 2 * result + value % 2
        value //= 2
    return result


def twiddle_table(configuration):
    return [
        pow(
            configuration["root"],
            bit_reverse(index, configuration["reversal_bits"]),
            configuration["modulus"],
        )
        for index in range(2 ** configuration["reversal_bits"])
    ]


def transform(values, twiddles, replacements, configuration):
    modulus = configuration["modulus"]
    state = [value % modulus for value in values]
    length = len(state) // 2
    twiddle_index = 1
    while length >= configuration["minimum_length"]:
        for start in range(0, len(state), 2 * length):
            twiddle = replacements.get(twiddle_index, twiddles[twiddle_index])
            for index in range(start, start + length):
                product = twiddle * state[index + length] % modulus
                first = state[index]
                state[index] = (first + product) % modulus
                state[index + length] = (first - product) % modulus
            twiddle_index += 1
        length //= 2
    return state


def difference_matrix(twiddles, replacements, configuration):
    columns = []
    for coordinate in range(DIMENSION):
        basis = [0] * DIMENSION
        basis[coordinate] = 1
        correct = transform(basis, twiddles, {}, configuration)
        faulted = transform(basis, twiddles, replacements, configuration)
        columns.append(
            [
                (correct[index] - faulted[index]) % configuration["modulus"]
                for index in range(DIMENSION)
            ]
        )
    return [list(row) for row in zip(*columns)]


def matrix_rank(rows, modulus):
    matrix = [row.copy() for row in rows]
    pivot = 0
    for column in range(DIMENSION):
        selected = next(
            (row for row in range(pivot, len(matrix)) if matrix[row][column]),
            None,
        )
        if selected is None:
            continue
        matrix[pivot], matrix[selected] = matrix[selected], matrix[pivot]
        inverse = pow(matrix[pivot][column], modulus - 2, modulus)
        matrix[pivot] = [value * inverse % modulus for value in matrix[pivot]]
        for row in range(pivot + 1, len(matrix)):
            multiplier = matrix[row][column]
            if multiplier:
                matrix[row] = [
                    (value - multiplier * leading) % modulus
                    for value, leading in zip(matrix[row], matrix[pivot])
                ]
        pivot += 1
        if pivot == len(matrix):
            break
    return pivot


def verify():
    results = []
    for configuration in CONFIGURATIONS:
        twiddles = twiddle_table(configuration)
        assert tuple(twiddles[1:4]) == configuration["first_three_twiddles"]
        for third, expected in (
            (GENERIC_REPLACEMENT, EXPECTED_GENERIC_RANK),
            (configuration["exceptional_replacement"], EXPECTED_EXCEPTIONAL_RANK),
        ):
            replacements = dict(zip(FAULT_INDICES, (1, 1, third)))
            assert all(replacements[index] != twiddles[index] for index in FAULT_INDICES)
            joint = difference_matrix(twiddles, replacements, configuration)
            stacked = []
            for index in FAULT_INDICES:
                stacked.extend(
                    difference_matrix(twiddles, {index: replacements[index]}, configuration)
                )
            joint_rank = matrix_rank(joint, configuration["modulus"])
            rowspace_rank = matrix_rank(stacked, configuration["modulus"])
            assert joint_rank == expected
            assert rowspace_rank == EXPECTED_ROWSPACE_RANK
            results.append(
                {
                    "scheme": configuration["scheme"],
                    "replacements": [replacements[index] for index in FAULT_INDICES],
                    "joint_rank": joint_rank,
                    "singleton_rowspace_dimension": rowspace_rank,
                    "joint_kernel_dimension": DIMENSION - joint_rank,
                    "singleton_kernel_intersection_dimension": DIMENSION - rowspace_rank,
                }
            )
    return {
        "configuration": {
            "dimension": DIMENSION,
            "fault_indices": FAULT_INDICES,
            "families": CONFIGURATIONS,
            "expected_generic_rank": EXPECTED_GENERIC_RANK,
            "expected_exceptional_rank": EXPECTED_EXCEPTIONAL_RANK,
            "expected_rowspace_rank": EXPECTED_ROWSPACE_RANK,
        },
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))