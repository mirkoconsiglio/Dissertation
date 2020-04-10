def test(data, **kwargs):
    def test2(data=None, **kwargs):
        if data == None:
            _data = {}
        else:
            _data = data

        for i in kwargs:
            _data[i] = kwargs[i]

        return _data

    q = 5
    updated_data = test2(data, q=q, **kwargs)
    return updated_data

data = test(data={'z': (1 + 1j), 'y': ['hello', 25]}, a=10, c='test')
print(data)