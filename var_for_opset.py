def var(g, input, *args):
    if args[0].type().isSubtypeOf(ListType.ofInts()):
        s = _std(g, input, *args)
    else:
        s = _std(g, input, None, args[0], None)
    return g.op('Mul', s, s)