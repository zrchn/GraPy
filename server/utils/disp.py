class bcolors:
    green = '\x1b[92m'
    yellow = '\x1b[93m'
    red = '\x1b[91m'
    RESET = '\x1b[0m'
    blue = '\x1b[96m'
    purple = '\x1b[95m'
    plain = '\x1b[37m'
    bold = '\x1b[97m'
    cyan = '\x1b[36m'

class Disp(bcolors):

    def __call__(self, text, *args, **kwargs):
        add = ''.join([' ' + a for a in args])
        color = getattr(self, kwargs.get('color', 'plain'), 'plain')
        print(color + str(text) + add + self.RESET, *args, **{k: v for k, v in kwargs.items() if not k == 'color'})
disp = Disp()
if __name__ == '__main__':
    disp = Disp()
    disp('hello, this is a message.', color='green')
    disp('this is a success.', color='yellow')
    disp('this is a warning.', color='red')
    disp('this is an error.', color='blue')
    disp('this is something interesting.', color='bold')
    disp('this is something fancy.', color='cyan')