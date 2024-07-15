def my_function(case):
    if case == 1:
        return 1
    elif case == 2:
        return 1, 2
    elif case == 3:
        return 1, 2, 3


# 调用函数并处理返回值
def handle_return_values(case):
    result = my_function(case)

    if isinstance(result, tuple):
        if len(result) == 1:
            val1 = result[0]
            print(f"Received 1 value: {result}")
        elif len(result) == 2:
            val1, val2 = result
            print(f"Received 2 values: {result}")
        elif len(result) == 3:
            val1, val2, val3 = result
            print(f"Received 3 values: {result}")
    else:
        val1 = result
        print(f"Received 0 value: {val1}")


# 示例调用
handle_return_values(1)
handle_return_values(2)
handle_return_values(3)