import lib

items = lib.Items()
x_elem = Element("x")
y_elem = Element("y")
label_elem = Element("label")


def clear_form():
    x_elem.clear()
    y_elem.clear()


def train_and_plot(*ags, **kws):
    _x0 = x_elem.element.value
    _x1 = y_elem.element.value
    _y = label_elem.element.value
    # add new item
    if len(_x0) > 0 and len(_x1) > 0 and len(_y) > 0:
        _x0, _x1, _y = list(map(float, [_x0, _x1, _y]))

        clear_form()
        items.append(_x0, _x1, _y)

    # train
    clf = lib.train(items.x, items.y)

    # predict
    y_pred = clf.predict(items.x)

    # plot
    fig = lib.plot(items.x, y_pred, clf.coef_[0], clf.intercept_[0])
    pyscript.write("figure1", fig)


train_and_plot()
