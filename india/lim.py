import json, cv2, numpy as np
tiles = []
for v, ids in (("cam_20261106_100001", [4378, 478, 859]), ("cam_20261106_100849", [666, 859, 2559, 165])):
    tr = {t["id"]: t for t in json.load(open(f"india_mine/{v}/tracks.json"))}
    for i in ids:
        if i not in tr: continue
        bs = sorted(tr[i]["boxes"], key=lambda fb: -(fb[1][2] - fb[1][0]) * (fb[1][3] - fb[1][1]))
        for f, b in bs[:2]:
            im = cv2.imread(f"india_mine/{v}/frames/f{f:06d}.jpg"); H, W = im.shape[:2]
            x0, y0, x1, y1 = map(int, b[:4]); m = 25
            t = cv2.resize(im[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)], (280, 280))
            cv2.putText(t, f"{v[-3:]} #{i}", (3, 18), 0, 0.55, (0, 255, 255), 2); tiles.append(t)
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
cv2.imwrite("review/limits.jpg", np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)]))
