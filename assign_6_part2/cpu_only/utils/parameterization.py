"""Aims to provide efficient way to control and generate experiment paramters.

"""

import itertools
import dataclasses


def iterate_dict(dictionary: dict):
    """This function iterates over combinations of dictionary values.

    Given a dictionary with elements as lists, this function generates
    single-element dictionary with all possible combinations of elements.

    Args:
        dictionary (dict): The input dictionary.
    """
    dict_keys = dictionary.keys()
    dict_values = dictionary.values()

    # convert single values to list
    dict_values = [
        [value] if not isinstance(value, list) else value for value in dict_values
    ]

    value_combinations = list(itertools.product(*dict_values))

    result = [
        dict(zip(dict_keys, value_combination))
        for value_combination in value_combinations
    ]
    return result


def recursive_iterate_dict(dictionary: dict):
    """This function iterates over combinations of dictionary values.

    Given a dictionary with elements as lists or dictionaries, this function
    generates single-element dictionary with all possible combinations of
    elements in the dictionary or its sub-dictionaries recursively.

    Args:
        dictionary (dict): The input dictionary.
    """

    def recursive_dict_combinations(subdict: dict):
        """This is the helper function of recursive_iterate_dict.

        This function recursively calls itself to iterate over all subsidiary
        dictionaries of the input dictionary and return all value combinations
        in the current subdict.
        """
        keys = []
        values = []

        for key, value in subdict.items():
            if isinstance(value, list):
                keys.append(key)
                values.append(value)
            elif isinstance(value, dict):
                sub_combinations = recursive_dict_combinations(value)
                keys.append(key)
                values.append(sub_combinations)
            else:
                # single value situation
                keys.append(key)
                values.append([value])

        combinations = itertools.product(*values)

        return [dict(zip(keys, combination)) for combination in combinations]

    return recursive_dict_combinations(dictionary)


def recursive_iterate_dataclass(dataclass_instance: dataclasses.dataclass):
    """This function iterates over combinations of dataclass values.

    Given a dataclass instance with elements as lists or dataclasses, this
    function generates single-element dictionary with all possible combinations
    of elements in the dataclass or its sub-dataclasses recursively.

    Args:
        dataclass_instance (dataclass): The input
    """

    def recursive_dataclass_combinations(current_instance: any):
        """This is the helper function of recursive_iterate_dataclass.

        This function recursively calls itself to iterate over all subsidiary
        dataclasses of the input dataclass and return all value combinations
        in the current dataclass.
        """

        # check if the current instance is a dataclass
        # only recursive iterate if it is a dataclass
        if not dataclasses.is_dataclass(current_instance):
            return [current_instance]

        keys = []
        values = []

        for fields in dataclasses.fields(current_instance):
            value = getattr(current_instance, fields.name)
            if isinstance(value, list):
                keys.append(fields.name)
                values.append(value)
            elif dataclasses.is_dataclass(value):
                sub_combinations = recursive_dataclass_combinations(value)
                keys.append(fields.name)
                values.append(sub_combinations)
            else:
                # single value situation
                keys.append(fields.name)
                values.append([value])

        # generate all combinations of values
        combinations = itertools.product(*values)

        # create a list of dataclass instances
        return [
            dataclasses.replace(current_instance, **dict(zip(keys, combination)))
            for combination in combinations
        ]

    return recursive_dataclass_combinations(dataclass_instance)


# provide a module test
if __name__ == "__main__":
    test_dict_with_subdict = {
        "a": [1, 2],
        "b": {"c": [3, 4], "d": [5, 6]},
    }
    test_dict = {
        "a": [1, 2],
        "b": [3, 4],
    }
    print(recursive_iterate_dict(test_dict_with_subdict))
    print(iterate_dict(test_dict))

    @dataclasses.dataclass
    class SubDataclass:
        c: list
        d: list

    @dataclasses.dataclass
    class TestDataclass:
        a: list
        b: SubDataclass

    test_dataclass = TestDataclass([1, 2], SubDataclass([3, 4], [5, 6]))

    # print all dataclass combinations
    for dataclass_instance in recursive_iterate_dataclass(test_dataclass):
        for field in dataclasses.fields(dataclass_instance):
            print(f"{field.name}: {getattr(dataclass_instance, field.name)}")
