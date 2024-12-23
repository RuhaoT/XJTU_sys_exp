import argparse
from numpy.random import uniform


def get_inputs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mat_size", type=int)
    args = argparser.parse_args()
    return args.mat_size


if __name__ == "__main__":
    mat_size = get_inputs()
    matrix_a = uniform(0.0, 1.0, mat_size * mat_size)
    matrix_b = uniform(0.0, 1.0, mat_size * mat_size)

    lines = ["double A [NUM_ELEMENTS] = { \n"]
    line = ""
    for element in matrix_a:
        printable = round(element, 2)
        if line == "":
            line = f"{printable}"
        else:
            line += f", {printable}"
        if len(line) > 80:
            line += ", \n"
            lines.append(line)
            line = ""
    if line != "":
        line += " \n"
        lines.append(line)
    lines.append("}; \n")
    lines.append("\n")

    lines.append("double B [NUM_ELEMENTS] = { \n")
    line = ""
    for element in matrix_b:
        printable = round(element, 2)
        if line == "":
            line = f"{printable}"
        else:
            line += f", {printable}"
        if len(line) > 80:
            line += ", \n"
            lines.append(line)
            line = ""
    if line != "":
        line += " \n"
        lines.append(line)
    lines.append("}; \n")
    lines.append("\n")
    lines.append("#endif // __MATMUL_MATRIX_H__\n")

    lines = [
        "#ifndef __MATMUL_MATRIX_H__\n",
        "#define __MATMUL_MATRIX_H__\n",
        "\n",
        f"#define SIZE {mat_size}\n",
        f"#define NUM_ELEMENTS {mat_size*mat_size}",
        "\n",
        f"double C [NUM_ELEMENTS] = {{0}};\n",
        "\n",
    ] + lines
    with open("matrix.h", "w") as header_file:
        header_file.writelines(lines)
