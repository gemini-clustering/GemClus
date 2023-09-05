import functools
from inspect import signature

from sklearn.utils import _param_validation as skparamvalid


class InvalidParameterError(ValueError, TypeError):
    pass


def check_constraint(local_constraint):
    if isinstance(local_constraint, str) and local_constraint == "array-like":
        return skparamvalid._ArrayLikes()
    if isinstance(local_constraint, str) and local_constraint == "random_state":
        return skparamvalid._RandomStates()
    if isinstance(local_constraint, str) and (local_constraint == "boolean" or local_constraint == "bool"):
        return skparamvalid._Booleans()
    if local_constraint is callable:
        return skparamvalid._Callables()
    if local_constraint is None:
        return skparamvalid._NoneConstraint()
    if isinstance(local_constraint, (skparamvalid.Interval,
                                     skparamvalid.StrOptions,
                                     skparamvalid.Options)):
        return local_constraint
    if isinstance(local_constraint, type):
        return skparamvalid._InstancesOf(local_constraint)
    raise ValueError(f"Unhandled constraint type: {local_constraint}")


# Credit: this snippet of code was strongly inspired and taken
# from scikit learn validate_params v1.2.2
def constraint_params(parameter_constraints):
    def decorate_with_constraints(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_signature = signature(func)

            # map arguments according to signature
            params = function_signature.bind(*args, **kwargs)
            params.apply_defaults()

            # We must ignore "self" arguments and positional arguments
            to_ignore = [
                p.name for p in function_signature.parameters.values()
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            to_ignore += ["self"]

            filtered_params = {k: v for k, v in params.arguments.items() if k not in to_ignore}

            # Verify that the arguments now satisfy correct constraints
            for param_name, param_val in filtered_params.items():
                if param_name not in parameter_constraints:
                    continue

                local_constraints = parameter_constraints[param_name]
                converted_constraints = [check_constraint(c) for c in local_constraints]

                is_satisfied = False
                for local_constraint in converted_constraints:
                    is_satisfied = is_satisfied or local_constraint.is_satisfied_by(param_val)
                    if is_satisfied:
                        break
                if not is_satisfied:
                    if len(local_constraints) == 1:
                        descriptor = f"{converted_constraints[0]}"
                    else:
                        descriptor = ", ".join([str(x) for x in converted_constraints[:-1]])
                        descriptor += f" or {converted_constraints[-1]}"
                    raise InvalidParameterError(f"The {param_name} parameter of {func.__name__} must "
                                                f"be {descriptor}. Got {param_val} instead.")

            # Call function with safe args
            return func(*args, **kwargs)

        return wrapper

    return decorate_with_constraints
