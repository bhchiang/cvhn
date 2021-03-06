d = {'a': 1, 'b': 2}


def test(a='a', b='b'):
    print(a, b)


test()
test(**d)
