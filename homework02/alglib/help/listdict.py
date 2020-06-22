# coding: UTF8


class PropDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]

        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def get_data(self, key):
        val = self[key]
        return val

    def get_data_up(self, key):
        val = self[key.upper()]
        return val


def all_attribute(self, dictionary):
    for key, value in dictionary.items():
        setattr(self, key, value)
