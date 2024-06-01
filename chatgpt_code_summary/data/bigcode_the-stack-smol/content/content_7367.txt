import json
import os
# set working directory


def gen_json(t):
    print(os.getcwd())
    # read log file
    with open('Screenshots/Screenshoot_meta.txt', 'r') as f:
        log = f.read()
        data = {"camera_angle_x": 0.6911112070083618}
        frames = []
        line_cnt = 0
        for line in log.split('\n'):
            try:
                record = {"file_path": "{}}_{:04d}".format(t,
                                                           line_cnt), "rotation": 4.0, "transform_matrix": eval(line)}
            except:
                pass
            frames.append(record)
            line_cnt += 1
        data["frames"] = frames

        data_json = json.dumps(data)
        with open('Screenshots/Screenshoot_meta.json', 'w') as ff:
            ff.write(data_json)
# %%
# %%
