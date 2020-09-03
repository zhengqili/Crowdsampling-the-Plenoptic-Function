
def create_model(opt, _isTrain):
    model = None
    from .render_model import RenderModel
    model = RenderModel(opt, _isTrain)
    print("model [%s] was created" % (model.name()))
    return model
