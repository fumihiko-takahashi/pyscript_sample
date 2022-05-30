import lib

items = lib.Items()
new_item_elem = Element("new-item-elem")

# 初期表示
fig = lib.plot(items.x, None, None, None)
pyscript.write("figure1", fig)


def add_item(*ags, **kws):
    new_item = new_item_elem.element.value
    if len(new_item) == 0:
        return None

    # new item に要素があれば実行
    _x0, _x1, _y = list(map(float, new_item.split(",")))
    new_item_elem.clear()
    items.append(_x0, _x1, _y)

    # train
    clf = lib.train(items.x, items.y)

    # predict
    y_pred = clf.predict(items.x)

    # plot
    fig = lib.plot(items.x, y_pred, clf.coef_[0], clf.intercept_[0])
    pyscript.write("figure1", fig)
