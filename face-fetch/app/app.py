from facefetch.main import populate_images, init_query_vector
import urllib
import json


def active_learn(handler):
    ''' Parse variables from the handler object and call populate images'''
    body_args = urllib.parse.parse_qs(handler.request.body.decode('utf-8'))
    keys = ['attrs[]', 'similar[]', 'dissimilar[]']
    inputs = [body_args.get(key, []) for key in keys]
    print(*inputs)
    # return populate_images(*inputs)
    test_images = [
        "CFD-AM-223-138-N.jpg",
        "CFD-BM-043-071-N.jpg",
        "CFD-WF-226-095-N.jpg",
        "CFD-LF-246-129-N.jpg",
        "CFD-LM-235-231-N.jpg",
        "CFD-LM-238-129-N.jpg",

    ]
    return json.dumps(test_images)


def reset_user(handler):
    init_query_vector()
    return 'True'
