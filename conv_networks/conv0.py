import util
import model
import params
import layers

def main():
    mnist = util.get_mnist()
    _params = params.get_params()
    _model = model.get_model(mnist, _params)

if __name__ == '__main__':
    main()
