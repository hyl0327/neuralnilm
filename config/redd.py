""" Config for the REDD dataset """


# Windows (training, validation, testing)
WINDOWS = {
    'train': {
        1: ('2011-04-01', '2011-05-01'),
        2: ('2011-04-01', '2011-05-01'),
    },
    'unseen_activations_of_seen_appliances': {
        1: ('2011-05-01', None),
        2: ('2011-05-01', None),
    },
    'unseen_appliances': {
        3: ('2011-04-01', None),
    }
}


# Appliances
APPLIANCES = [
    'kettle',
    'microwave',
    'washing machine',
    'dish washer',
    'fridge',
]


# Training & validation buildings, and testing buildings
BUILDINGS = {
    'kettle': {
        'train_buildings': [1, 2],
        'unseen_buildings': [3],
    },
    'microwave': {
        'train_buildings': [1, 2],
        'unseen_buildings': [3],
    },
    'washing machine': {
        'train_buildings': [1, 2],
        'unseen_buildings': [3],
    },
    'fridge': {
        'train_buildings': [1, 2],
        'unseen_buildings': [3],
    },
    'dish washer': {
        'train_buildings': [1, 2],
        'unseen_buildings': [3],
    },
}
