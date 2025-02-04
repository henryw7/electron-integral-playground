
import sympy
import re

def simplify(term: str, if_remove_power: bool = True) -> str:
    term = sympy.simplify(term)

    if if_remove_power:
        power_operations = list(term.atoms(sympy.Pow))

        if False:
            if any(not exponent.is_Integer for base, exponent in (power_operation.as_base_exp() for power_operation in power_operations)):
                raise ValueError("A power contains a non-integer exponent")
            replace_map = zip(power_operations, (sympy.Mul(*[base]*exponent, evaluate = False) for base,exponent in (power_operation.as_base_exp() for power_operation in power_operations)))
            term = term.subs(replace_map, evaluate = False)
        # The following code is applying the sympy logic above using regex.
        # Why not sympy? Because it doesn't work. Sympy cannot resist to evaluate the expression, and change multiplications to powers.

        power_operations = [str(power_operation) for power_operation in power_operations]

        replacement_map = {}
        not_fully_replaced_list = []
        for power_operation in power_operations:
            match_result = re.fullmatch(r"(\(.+\)|.+)\s*\*\*\s*(\d+)", power_operation)
            assert match_result is not None
            base = match_result[1]
            exponent = int(match_result[2])
            assert exponent > 1
            if "**" not in base:
                replacement_map[power_operation] = "*".join([base] * exponent)
            else:
                not_fully_replaced_list.append([power_operation, base, exponent])

        n_not_fully_replaced = len(not_fully_replaced_list)
        for i_replace in range(n_not_fully_replaced**2):
            full_expression, base, exponent = not_fully_replaced_list.pop()
            full_expression_replaced = full_expression
            for expression_to_replace in replacement_map:
                base = base.replace(expression_to_replace, replacement_map[expression_to_replace])
                full_expression_replaced = full_expression_replaced.replace(expression_to_replace, replacement_map[expression_to_replace])
            if "**" not in base:
                processed_expression = "*".join([base] * exponent)
                replacement_map[full_expression] = processed_expression
                replacement_map[full_expression_replaced] = processed_expression
            else:
                not_fully_replaced_list.insert(0, [full_expression, base, exponent])

            if not not_fully_replaced_list:
                break
            if i_replace == n_not_fully_replaced**2 - 1:
                raise ValueError(f"Expressions cannot be simplified: {not_fully_replaced_list}")

        term = str(term)
        for expression_to_replace in replacement_map:
            term = term.replace(expression_to_replace, replacement_map[expression_to_replace])
        assert "**" not in term
    else:
        term = str(term)

    return term

def remove_whitespace(line: str) -> str:
    return "".join(line.split())

def sympy_number_to_cpp_float(number: sympy.Expr) -> str:
    assert len(number.free_symbols) == 0
    # if number.is_integer:
    #     return str(number)
    if number == 0:
        return "0"
    number = str(number)
    assert "." not in number
    number = re.sub(r"(\d+)", r"\1.0", number)
    return number
