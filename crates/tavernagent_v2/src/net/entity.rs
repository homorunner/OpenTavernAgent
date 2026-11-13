//! Shared encoders for observable entities and optional object-conditioned action heads.
//! Slot order is retained. No pooling, hidden game state or extra observations are used.
use super::*;

const WIDTH: usize = 64;
const GLOBAL_WIDTH: usize = 128;
const GROUPS: usize = 6;
const ENCODER_TENSORS: usize = (GROUPS + 1) * 4;
const ACTION_STARTS: [usize; 5] = [a::A_BUY, a::A_SELL, a::A_ACTIVATE, a::A_CHOOSE, a::A_PLAY];
const ACTION_COUNTS: [usize; 5] = [
    a::N_SHOP,
    a::N_BOARD,
    a::N_BOARD,
    a::N_CHOICE,
    a::N_HAND * crate::actions::N_POS,
];

pub(super) fn tensor_names() -> Vec<String> {
    [
        "meta", "board", "hand", "shop", "choice", "opponent", "global", "buy", "sell", "activate",
        "choose", "play",
    ]
    .iter()
    .flat_map(|name| {
        [
            format!("W_{name}_in"),
            format!("b_{name}_in"),
            format!("W_{name}_out"),
            format!("b_{name}_out"),
        ]
    })
    .collect()
}

#[derive(Clone, Debug)]
struct Group {
    slots: Vec<usize>,
    features: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub(super) struct Layout {
    groups: Vec<Group>,
    global: Vec<usize>,
    numeric: usize,
    slots: usize,
    embedding: usize,
}

struct MlpCache {
    input: Mat,
    derivative: Mat,
    hidden: Mat,
}
pub(super) struct Cache {
    groups: Vec<MlpCache>,
    present: Vec<Vec<bool>>,
    global: MlpCache,
}
pub(super) struct ActionCache {
    head: Mat,
    groups: Vec<MlpCache>,
    boards: Vec<Mat>,
}

fn mlp(net: &PolicyValueNet, index: usize, input: Mat) -> (Mat, MlpCache) {
    let p = &net.p.tensors;
    let mut pre = linear(&input, &p[index]);
    pre.add_row(&p[index + 1].data);
    let (hidden, derivative) = activate_with_derivative(pre, net.cfg.activation);
    let mut out = linear(&hidden, &p[index + 2]);
    out.add_row(&p[index + 3].data);
    (
        out,
        MlpCache {
            input,
            derivative,
            hidden,
        },
    )
}

fn mlp_backward(
    net: &PolicyValueNet,
    index: usize,
    cache: &MlpCache,
    dout: &Mat,
    grads: &mut Params,
) -> Mat {
    let p = &net.p.tensors;
    grads.tensors[index + 2] = cache.hidden.t_matmul(dout);
    grads.tensors[index + 3] = Mat::from_vec(1, dout.cols, dout.sum_rows());
    let mut dh = dout.matmul_t(&p[index + 2]);
    for (d, x) in dh.data.iter_mut().zip(&cache.derivative.data) {
        *d *= x;
    }
    grads.tensors[index] = cache.input.t_matmul(&dh);
    grads.tensors[index + 1] = Mat::from_vec(1, dh.cols, dh.sum_rows());
    dh.matmul_t(&p[index])
}

/// Scratch only: no parameters or observation-dependent values survive a call
/// without being overwritten. Shared across groups and across rollout decisions.
#[derive(Default)]
pub(super) struct InferenceWorkspace {
    input: Mat,
    hidden: Mat,
    out: Mat,
    board: Mat,
    global_projection: Mat,
    object_projection: Mat,
    board_projection: Mat,
}

fn mlp_infer(net: &PolicyValueNet, index: usize, ws: &mut InferenceWorkspace) {
    let p = &net.p.tensors;
    linear_into(&ws.input, &p[index], &mut ws.hidden);
    ws.hidden.add_row(&p[index + 1].data);
    for x in &mut ws.hidden.data {
        *x = net.cfg.activation.apply(*x);
    }
    linear_into(&ws.hidden, &p[index + 2], &mut ws.out);
    ws.out.add_row(&p[index + 3].data);
}

impl Layout {
    pub(super) fn new(features: &[usize], cards: &[usize], embedding: usize) -> Self {
        let mut groups: Vec<Group> = (0..GROUPS)
            .map(|_| Group {
                slots: vec![],
                features: vec![],
            })
            .collect();
        let board_card = 1 + a::N_POWERS + 2 + 3 * o::N_QUESTS + o::N_SECRETS;
        let board_float = o::GLOBAL_F + o::N_QUESTS * o::QUEST_F + o::PENDING_KINDS;
        let mut card_start = board_card;
        let mut float_start = board_float;
        for (kind, count, width) in [
            (1, a::N_BOARD, o::MINION_F),
            (2, a::N_HAND, o::ITEM_F),
            (3, a::N_SHOP, o::ITEM_F),
            (4, a::N_CHOICE, o::ITEM_F),
            (5, o::N_OPP, o::OPP_F),
        ] {
            if kind == 5 {
                card_start += 8;
            }
            for slot in 0..count {
                groups[kind].slots.push(
                    cards
                        .iter()
                        .position(|&c| c == card_start + slot)
                        .expect("entity card omitted"),
                );
                groups[kind].features.push(
                    (0..width)
                        .map(|i| {
                            ONE_HOT_FEATURES
                                + features
                                    .iter()
                                    .position(|&f| f == float_start + slot * width + i)
                                    .expect("entity feature omitted")
                        })
                        .collect(),
                );
            }
            card_start += count;
            float_start += count * width;
        }
        for slot in 0..cards.len() {
            if !groups.iter().any(|g| g.slots.contains(&slot)) {
                groups[0].slots.push(slot);
                groups[0].features.push(vec![]);
            }
        }
        let numeric = ONE_HOT_FEATURES + features.len();
        let global = (0..numeric)
            .filter(|f| {
                !groups
                    .iter()
                    .any(|g| g.features.iter().any(|fs| fs.contains(f)))
            })
            .collect();
        Self {
            groups,
            global,
            numeric,
            slots: cards.len(),
            embedding,
        }
    }

    pub(super) fn output_dim(&self) -> usize {
        self.slots * WIDTH + GLOBAL_WIDTH + N_ACTIONS
    }
    fn raw_dim(&self) -> usize {
        self.numeric + self.slots * self.embedding + N_ACTIONS
    }

    pub(super) fn initialize(
        &self,
        tensors: &mut Vec<Mat>,
        cfg: &NetConfig,
        randn: &mut impl FnMut(usize, usize, f32) -> Mat,
    ) {
        let mut add = |input: usize, hidden: usize, output: usize, scale: f32| {
            tensors.push(randn(input, hidden, (2.0 / input as f32).sqrt()));
            tensors.push(Mat::zeros(1, hidden));
            tensors.push(randn(hidden, output, (2.0 / hidden as f32).sqrt() * scale));
            tensors.push(Mat::zeros(1, output));
        };
        for group in &self.groups {
            add(self.embedding + group.features[0].len(), WIDTH, WIDTH, 1.0);
        }
        add(self.global.len(), GLOBAL_WIDTH, GLOBAL_WIDTH, 1.0);
        {
            for kind in 0..5 {
                add(Self::action_dim(cfg.hidden, kind), WIDTH, 1, 0.1);
            }
        }
    }

    pub(super) fn infer(
        &self,
        net: &PolicyValueNet,
        raw: &Mat,
        used: &[i32],
        output: &mut Mat,
        ws: &mut InferenceWorkspace,
    ) {
        output.resize(raw.rows, self.output_dim());
        output.data.fill(0.0);
        for (kind, group) in self.groups.iter().enumerate() {
            let count = group.slots.len();
            ws.input
                .resize(raw.rows * count, self.embedding + group.features[0].len());
            for r in 0..raw.rows {
                for (j, &slot) in group.slots.iter().enumerate() {
                    let row = ws.input.row_mut(r * count + j);
                    let start = self.numeric + slot * self.embedding;
                    row[..self.embedding]
                        .copy_from_slice(&raw.row(r)[start..start + self.embedding]);
                    for (v, &f) in row[self.embedding..].iter_mut().zip(&group.features[j]) {
                        *v = raw.row(r)[f];
                    }
                }
            }
            mlp_infer(net, net.p.extra() + kind * 4, ws);
            for r in 0..raw.rows {
                for (j, &slot) in group.slots.iter().enumerate() {
                    let present = if kind == 0 {
                        used[r * self.slots + slot] != 0
                    } else {
                        raw.row(r)[group.features[j][0]] != 0.0
                    };
                    if present {
                        output.row_mut(r)[slot * WIDTH..(slot + 1) * WIDTH]
                            .copy_from_slice(ws.out.row(r * count + j));
                    }
                }
            }
        }
        ws.input.resize(raw.rows, self.global.len());
        for r in 0..raw.rows {
            for (v, &f) in ws.input.row_mut(r).iter_mut().zip(&self.global) {
                *v = raw.row(r)[f];
            }
        }
        mlp_infer(net, net.p.extra() + GROUPS * 4, ws);
        for r in 0..raw.rows {
            let start = self.slots * WIDTH;
            output.row_mut(r)[start..start + GLOBAL_WIDTH].copy_from_slice(ws.out.row(r));
            output.row_mut(r)[start + GLOBAL_WIDTH..]
                .copy_from_slice(&raw.row(r)[self.raw_dim() - N_ACTIONS..]);
        }
    }

    pub(super) fn infer_policy(
        &self,
        net: &PolicyValueNet,
        head: &Mat,
        encoded: &Mat,
        logits: &mut Vec<f32>,
        ws: &mut InferenceWorkspace,
    ) {
        assert_eq!(head.rows, 1);
        logits.resize(N_ACTIONS, 0.0);
        let positions = crate::actions::N_POS;
        for (kind, &count) in ACTION_COUNTS.iter().enumerate() {
            let index = net.p.extra() + ENCODER_TENSORS + kind * 4;
            let p = &net.p.tensors;
            let play = kind == 4;
            let objects = if play { a::N_HAND } else { count };
            ws.input.resize(objects, WIDTH);
            for j in 0..objects {
                let (slot, _) = self.action_slots(kind, if play { j * positions } else { j });
                ws.input
                    .row_mut(j)
                    .copy_from_slice(&encoded.row(0)[slot * WIDTH..(slot + 1) * WIDTH]);
            }
            head.matmul_weight_rows_into(&p[index], 0, head.cols, false, &mut ws.global_projection);
            ws.input.matmul_weight_rows_into(
                &p[index],
                head.cols,
                WIDTH,
                false,
                &mut ws.object_projection,
            );
            if play {
                ws.board.resize(positions, WIDTH);
                for j in 0..positions {
                    let slot = self.groups[1].slots[j];
                    ws.board
                        .row_mut(j)
                        .copy_from_slice(&encoded.row(0)[slot * WIDTH..(slot + 1) * WIDTH]);
                }
                ws.board.matmul_weight_rows_into(
                    &p[index],
                    head.cols + WIDTH,
                    WIDTH,
                    false,
                    &mut ws.board_projection,
                );
            }
            ws.hidden.resize(count, WIDTH);
            for j in 0..count {
                let obj = if play { j / positions } else { j };
                for (k, value) in ws.hidden.row_mut(j).iter_mut().enumerate() {
                    *value = ws.global_projection.row(0)[k] + ws.object_projection.row(obj)[k];
                    if play {
                        *value += ws.board_projection.row(j % positions)[k];
                        *value += p[index].row(head.cols + WIDTH * 2 + j % positions)[k];
                    }
                    *value += p[index + 1].data[k];
                    *value = net.cfg.activation.apply(*value);
                }
            }
            linear_into(&ws.hidden, &p[index + 2], &mut ws.out);
            for j in 0..count {
                logits[ACTION_STARTS[kind] + j] = ws.out.data[j] + p[index + 3].data[0];
            }
        }
    }

    pub(super) fn forward(&self, net: &PolicyValueNet, raw: &Mat, used: &[i32]) -> (Mat, Cache) {
        let mut output = Mat::zeros(raw.rows, self.output_dim());
        let mut caches = Vec::with_capacity(GROUPS);
        let mut present = Vec::with_capacity(GROUPS);
        for (kind, group) in self.groups.iter().enumerate() {
            let count = group.slots.len();
            let mut input = Mat::zeros(raw.rows * count, self.embedding + group.features[0].len());
            let mut valid = vec![false; input.rows];
            for r in 0..raw.rows {
                for (j, &slot) in group.slots.iter().enumerate() {
                    let row = input.row_mut(r * count + j);
                    let start = self.numeric + slot * self.embedding;
                    row[..self.embedding]
                        .copy_from_slice(&raw.row(r)[start..start + self.embedding]);
                    for (v, &f) in row[self.embedding..].iter_mut().zip(&group.features[j]) {
                        *v = raw.row(r)[f];
                    }
                    valid[r * count + j] = if kind == 0 {
                        used[r * self.slots + slot] != 0
                    } else {
                        raw.row(r)[group.features[j][0]] != 0.0
                    };
                }
            }
            let (encoded, cache) = mlp(net, net.p.extra() + kind * 4, input);
            for r in 0..raw.rows {
                for (j, &slot) in group.slots.iter().enumerate() {
                    if valid[r * count + j] {
                        output.row_mut(r)[slot * WIDTH..(slot + 1) * WIDTH]
                            .copy_from_slice(encoded.row(r * count + j));
                    }
                }
            }
            caches.push(cache);
            present.push(valid);
        }
        let mut globals = Mat::zeros(raw.rows, self.global.len());
        for r in 0..raw.rows {
            for (v, &f) in globals.row_mut(r).iter_mut().zip(&self.global) {
                *v = raw.row(r)[f];
            }
        }
        let (global, global_cache) = mlp(net, net.p.extra() + GROUPS * 4, globals);
        for r in 0..raw.rows {
            let start = self.slots * WIDTH;
            output.row_mut(r)[start..start + GLOBAL_WIDTH].copy_from_slice(global.row(r));
            output.row_mut(r)[start + GLOBAL_WIDTH..]
                .copy_from_slice(&raw.row(r)[self.raw_dim() - N_ACTIONS..]);
        }
        (
            output,
            Cache {
                groups: caches,
                present,
                global: global_cache,
            },
        )
    }

    pub(super) fn backward(
        &self,
        net: &PolicyValueNet,
        cache: &Cache,
        dout: &Mat,
        grads: &mut Params,
    ) -> Mat {
        let mut raw = Mat::zeros(dout.rows, self.raw_dim());
        for (kind, group) in self.groups.iter().enumerate() {
            let count = group.slots.len();
            let mut grad = Mat::zeros(dout.rows * count, WIDTH);
            for r in 0..dout.rows {
                for (j, &slot) in group.slots.iter().enumerate() {
                    if cache.present[kind][r * count + j] {
                        grad.row_mut(r * count + j)
                            .copy_from_slice(&dout.row(r)[slot * WIDTH..(slot + 1) * WIDTH]);
                    }
                }
            }
            let dx = mlp_backward(
                net,
                net.p.extra() + kind * 4,
                &cache.groups[kind],
                &grad,
                grads,
            );
            for r in 0..dout.rows {
                for (j, &slot) in group.slots.iter().enumerate() {
                    let start = self.numeric + slot * self.embedding;
                    raw.row_mut(r)[start..start + self.embedding]
                        .copy_from_slice(&dx.row(r * count + j)[..self.embedding]);
                }
            }
        }
        let mut grad = Mat::zeros(dout.rows, GLOBAL_WIDTH);
        for r in 0..dout.rows {
            grad.row_mut(r).copy_from_slice(
                &dout.row(r)[self.slots * WIDTH..self.slots * WIDTH + GLOBAL_WIDTH],
            );
        }
        // Global inputs are observation scalars, with no trainable embedding to scatter.
        mlp_backward(net, net.p.extra() + GROUPS * 4, &cache.global, &grad, grads);
        raw
    }

    fn action_dim(hidden: usize, kind: usize) -> usize {
        hidden
            + WIDTH
            + if kind == 4 {
                WIDTH + crate::actions::N_POS
            } else {
                0
            }
    }
    fn action_slots(&self, kind: usize, j: usize) -> (usize, Option<usize>) {
        match kind {
            0 => (self.groups[3].slots[j], None),
            1 | 2 => (self.groups[1].slots[j], None),
            3 => (self.groups[4].slots[j], None),
            4 => (
                self.groups[2].slots[j / crate::actions::N_POS],
                Some(self.groups[1].slots[j % crate::actions::N_POS]),
            ),
            _ => unreachable!(),
        }
    }

    // Each head partitions its existing first-layer matrix: [global; object;
    // board; position]. Linearity lets us project each distinct input once.
    pub(super) fn policy(
        &self,
        net: &PolicyValueNet,
        head: &Mat,
        encoded: &Mat,
        globals: &Mat,
    ) -> (Mat, ActionCache) {
        let mut logits = Mat::zeros(head.rows, N_ACTIONS);
        for r in 0..head.rows {
            logits.row_mut(r)[..a::A_BUY].copy_from_slice(globals.row(r));
        }
        let mut groups = Vec::with_capacity(5);
        let mut boards = Vec::with_capacity(5);
        let positions = crate::actions::N_POS;
        for (kind, &count) in ACTION_COUNTS.iter().enumerate() {
            let index = net.p.extra() + ENCODER_TENSORS + kind * 4;
            let p = &net.p.tensors;
            let play = kind == 4;
            let objects = if play { a::N_HAND } else { count };
            let mut input = Mat::zeros(head.rows * objects, WIDTH);
            let mut board = Mat::zeros(if play { head.rows * positions } else { 0 }, WIDTH);
            for r in 0..head.rows {
                for j in 0..objects {
                    let (slot, _) = self.action_slots(kind, if play { j * positions } else { j });
                    input
                        .row_mut(r * objects + j)
                        .copy_from_slice(&encoded.row(r)[slot * WIDTH..(slot + 1) * WIDTH]);
                }
                if play {
                    for j in 0..positions {
                        let slot = self.groups[1].slots[j];
                        board
                            .row_mut(r * positions + j)
                            .copy_from_slice(&encoded.row(r)[slot * WIDTH..(slot + 1) * WIDTH]);
                    }
                }
            }
            let global = head.matmul_weight_rows(&p[index], 0, head.cols, false);
            let object = input.matmul_weight_rows(&p[index], head.cols, WIDTH, false);
            let board_projection = if play {
                Some(board.matmul_weight_rows(&p[index], head.cols + WIDTH, WIDTH, false))
            } else {
                None
            };
            let mut pre = Mat::zeros(head.rows * count, WIDTH);
            for r in 0..head.rows {
                for j in 0..count {
                    let obj = if play { j / positions } else { j };
                    let row = pre.row_mut(r * count + j);
                    for (k, value) in row.iter_mut().enumerate() {
                        *value = global.row(r)[k] + object.row(r * objects + obj)[k];
                        if let Some(bp) = &board_projection {
                            *value += bp.row(r * positions + j % positions)[k];
                            *value += p[index].row(head.cols + WIDTH * 2 + j % positions)[k];
                        }
                        *value += p[index + 1].data[k];
                    }
                }
            }
            let (hidden, derivative) = activate_with_derivative(pre, net.cfg.activation);
            let out = linear(&hidden, &p[index + 2]);
            for r in 0..head.rows {
                for j in 0..count {
                    logits.row_mut(r)[ACTION_STARTS[kind] + j] =
                        out.data[r * count + j] + p[index + 3].data[0];
                }
            }
            groups.push(MlpCache {
                input,
                derivative,
                hidden,
            });
            boards.push(board);
        }
        (
            logits,
            ActionCache {
                head: head.clone(),
                groups,
                boards,
            },
        )
    }

    pub(super) fn policy_backward(
        &self,
        net: &PolicyValueNet,
        cache: &ActionCache,
        dlogits: &Mat,
        grads: &mut Params,
        dh: &mut Mat,
        dx: &mut Mat,
    ) -> Mat {
        let mut globals = Mat::zeros(dlogits.rows, a::A_BUY);
        for r in 0..dlogits.rows {
            globals
                .row_mut(r)
                .copy_from_slice(&dlogits.row(r)[..a::A_BUY]);
        }
        let positions = crate::actions::N_POS;
        for (kind, &count) in ACTION_COUNTS.iter().enumerate() {
            let index = net.p.extra() + ENCODER_TENSORS + kind * 4;
            let p = &net.p.tensors;
            let c = &cache.groups[kind];
            let play = kind == 4;
            let objects = if play { a::N_HAND } else { count };
            let mut grad = Mat::zeros(dlogits.rows * count, 1);
            for r in 0..dlogits.rows {
                for j in 0..count {
                    grad.data[r * count + j] = dlogits.row(r)[ACTION_STARTS[kind] + j];
                }
            }
            grads.tensors[index + 2] = c.hidden.t_matmul(&grad);
            grads.tensors[index + 3] = Mat::from_vec(1, 1, grad.sum_rows());
            let mut dp = grad.matmul_t(&p[index + 2]);
            for (d, x) in dp.data.iter_mut().zip(&c.derivative.data) {
                *d *= x;
            }
            grads.tensors[index + 1] = Mat::from_vec(1, WIDTH, dp.sum_rows());
            let mut dg = Mat::zeros(dlogits.rows, WIDTH);
            let mut dobj = Mat::zeros(dlogits.rows * objects, WIDTH);
            let mut db = Mat::zeros(if play { dlogits.rows * positions } else { 0 }, WIDTH);
            let mut dw = p[index].zeros_like();
            for r in 0..dlogits.rows {
                for j in 0..count {
                    let obj = if play { j / positions } else { j };
                    for (k, &d) in dp.row(r * count + j).iter().enumerate() {
                        dg.row_mut(r)[k] += d;
                        dobj.row_mut(r * objects + obj)[k] += d;
                        if play {
                            db.row_mut(r * positions + j % positions)[k] += d;
                            dw.row_mut(dh.cols + WIDTH * 2 + j % positions)[k] += d;
                        }
                    }
                }
            }
            let wg = cache.head.t_matmul(&dg);
            let wo = c.input.t_matmul(&dobj);
            dw.data[..dh.cols * WIDTH].copy_from_slice(&wg.data);
            dw.data[dh.cols * WIDTH..(dh.cols + WIDTH) * WIDTH].copy_from_slice(&wo.data);
            dh.add_assign(&dg.matmul_weight_rows(&p[index], 0, dh.cols, true));
            let di = dobj.matmul_weight_rows(&p[index], dh.cols, WIDTH, true);
            for r in 0..dlogits.rows {
                for j in 0..objects {
                    let (slot, _) = self.action_slots(kind, if play { j * positions } else { j });
                    for (d, x) in dx.row_mut(r)[slot * WIDTH..(slot + 1) * WIDTH]
                        .iter_mut()
                        .zip(di.row(r * objects + j))
                    {
                        *d += x;
                    }
                }
            }
            if play {
                let wb = cache.boards[kind].t_matmul(&db);
                dw.data[(dh.cols + WIDTH) * WIDTH..(dh.cols + WIDTH * 2) * WIDTH]
                    .copy_from_slice(&wb.data);
                let di = db.matmul_weight_rows(&p[index], dh.cols + WIDTH, WIDTH, true);
                for r in 0..dlogits.rows {
                    for j in 0..positions {
                        let slot = self.groups[1].slots[j];
                        for (d, x) in dx.row_mut(r)[slot * WIDTH..(slot + 1) * WIDTH]
                            .iter_mut()
                            .zip(di.row(r * positions + j))
                        {
                            *d += x;
                        }
                    }
                }
            }
            grads.tensors[index] = dw;
        }
        globals
    }
}

#[cfg(test)]
#[path = "entity_reference.rs"]
mod reference;
