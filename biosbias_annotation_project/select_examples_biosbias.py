import pandas as pd

targeted_occupation = 'physician'  # 'surgeon', 'nurse', 'dentist', 'physician', 'psychologist'
df = pd.read_csv("../data/biosbias_confident.csv")
# df = pd.read_csv("./data/biosbias_not_confident.csv")

print('Number of samples', len(df))

# Save examples in JSONL format for annotation
# with open('./data/biosbias_confident.jsonl', 'w') as f:
# psychologist_selection = [210, 212, 215, 216, 217, 221, 222, 229, 233, 236, 237, 239, 240, 245, 248, 255, 256, 266,
#                           268, 273, 279, 283, 290, 299, 316]
psychologist_selection = [210, 212, 215, 216, 229, 233, 236, 237, 239, 240, 245, 248, 255, 256, 266,
                          268, 273, 279, 283, 290, 299, 316]
not_confident = [3, 12, 14, 18, 22, 23, 24, 32, 34, 36, 38, 44, 49, 53, 70, 76, 78, 98]
# nurse_selection = [78, 79, 82, 86, 87, 89, 92, 95, 102, 109, 113, 119, 120, 127, 613, 617, 619, 623, 624, 626, 632,
#                    635, 634, 640, 651]
nurse_selection = [78, 79, 82, 86, 87, 89, 92, 95, 102, 109, 113, 119, 120, 613, 617, 619, 623, 624, 626, 632,
                   635, 634, 640, 651]
physician_selection = [131, 132, 134, 135, 136, 139, 154, 155, 165, 196, 200]
surgeon_selection = [444, 446, 447, 448, 449, 451, 453, 455, 457, 459, 461, 462, 463, 472, 474, 475, 482]
dentist_selection = [0, 1, 2, 3, 5, 9, 10, 12, 13, 14, 15, 16, 25, 62, 547, 597]

overlap_subset = [10, 229, 236, 273, 455]

confident_foils = [(0, 'Nurse'), (1, 'Nurse'), (2, 'Nurse'), (3, 'Nurse'), (5, 'Physician'), (9, 'Nurse'), (10, 'Nurse'),
             (12, 'Nurse'), (13, 'Nurse'), (14, 'Nurse'), (15, 'Nurse'), (16, 'Physician'), (25, 'Surgeon'),
             (62, 'Surgeon'), (78, 'Surgeon'), (79, 'Psychologist'), (82, 'Physician'), (86, 'Physician'),
             (87, 'Psychologist'), (89, 'Physician'), (92, 'Physician'), (95, 'Psychologist'), (102, 'Surgeon'),
             (109, 'Physician'), (113, 'Surgeon'), (119, 'Physician'), (120, 'Psychologist'),  (131, 'Nurse'),
             (132, 'Nurse'), (134, 'Nurse'), (135, 'Surgeon'), (136, 'Nurse'), (139, 'Psychologist'), (154, 'Psychologist'),
             (155, 'Surgeon'), (165, 'Surgeon'), (196, 'Nurse'), (200, 'Nurse'), (210, 'Nurse'), (212, 'Nurse'),
             (215, 'Physician'), (216, 'Nurse'), (229, 'Nurse'), (233, 'Nurse'), (236, 'Nurse'), (237, 'Nurse'),
             (239, 'Nurse'), (240, 'Nurse'), (245, 'Physician'), (248, 'Nurse'), (255, 'Physician'), (256, 'Nurse'),
             (266, 'Nurse'), (268, 'Nurse'), (273, 'Nurse'), (279, 'Physician'), (283, 'Physician'), (290, 'Physician'),
             (299, 'Nurse'), (316, 'Nurse'), (444, 'Physician'), (446, 'Physician'), (447, 'Physician'), (448, 'Physician'),
             (449, 'Physician'), (451, 'Physician'), (453, 'Physician'), (455, 'Physician'), (457, 'Physician'),
             (459, 'Physician'), (461, 'Physician'), (462, 'Physician'), (463, 'Physician'), (472, 'Physician'),
             (474, 'Physician'), (475, 'Physician'), (482, 'Dentist'), (547, 'Surgeon'), (597, 'Surgeon'),
             (613, 'Physician'), (617, 'Physician'), (619, 'Physician'), (623, 'Physician'), (624, 'Physician'),
             (626, 'Physician'), (632, 'Physician'), (634, 'Physician'), (635, 'Psychologist'), (640, 'Physician'), (651, 'Surgeon')]

not_confident_foils = [(3, 'Surgeon'), (12, 'Psychologist'), (14, 'Physician'), (18, 'Psychologist'), (22, 'Psychologist'),
                 (23, 'Surgeon'), (24, 'Nurse'), (32, 'Nurse'), (34, 'Nurse'), (36, 'Surgeon'), (38, 'Nurse'),
                 (44, 'Psychologist'), (49, 'Physician'), (53, 'Physician'), (70, 'Physician'), (76, 'Psychologist'),
                 (78, 'Psychologist'), (98, 'Physician')]

overlap_foils = [(10, 'Nurse'), (229, 'Physician'), (236, 'Physician'), (273, 'Nurse'), (455, 'Nurse')]

confident_foils = dict(confident_foils)
not_confident_foils = dict(not_confident_foils)

for irow, row in df.iterrows():
    # if targeted_occupation in row['1st prediction'] and irow in physician_selection:
    if irow in psychologist_selection + nurse_selection + physician_selection + surgeon_selection + dentist_selection:
        print('-' * 80)
        row['text'] = f'Why is the person in the following short bio described as a "{row["1st prediction"].title()}" rather than a "{row["2nd best prediction"].title()}" or a "{confident_foils[irow]}"?\n\n' \
                      + f'Bio: {row["text"]}'
        print('-' * 80)
        print('Example - ', irow)
        print(row['text'])
        print('-' * 80)
        input("Press Enter to continue...")
    if irow in not_confident:
        print('-' * 80)
        row[
            'text'] = f'Why is the person in the following short bio described as a "{row["1st prediction"].title()}" rather than a "{row["2nd best prediction"].title()}" or a "{not_confident_foils[irow]}"?\n\n' \
                      + f'Bio: {row["text"]}'
        print('-' * 80)
        print('Example - ', irow)
        print(row['text'])
        print('-' * 80)
        input("Press Enter to continue...")

        # f.write(row.to_json() + '\n')

