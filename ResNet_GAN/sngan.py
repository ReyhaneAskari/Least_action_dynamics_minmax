import os

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

import models.sngan as models
import common.losses as losses
from common.utils import generate_imgs, infiniteloop, set_seed
from common.score.score import get_inception_and_fid_score
from torch.nn.utils import parameters_to_vector

from old_adam import Adam

def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def parameters_grad_to_vector(parameters):
    param_device = None

    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        vec.append(param.grad.view(-1))
    return torch.cat(vec)


net_G_models = {
    'res32': models.ResGenerator32,
    'res48': models.ResGenerator48,
    'cnn32': models.Generator32,
    'cnn48': models.Generator48,
}

net_D_models = {
    'res32': models.ResDiscriminator32,
    'res48': models.ResDiscriminator48,
    'cnn32': models.Discriminator32,
    'cnn48': models.Discriminator48,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}


FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'stl10'], "dataset")
flags.DEFINE_enum('arch', 'res32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 100000, "total number of training steps")
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_float('reg', 0.0, "ODE_GAN regularization coef")
flags.DEFINE_float('lead_alpha_dis', 0.0, "lead_alpha_dis")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/SNGAN_CIFAR10_RES', 'log folder')
flags.DEFINE_bool('record', True, "record inception score and FID score")
flags.DEFINE_string('fid_cache', './stats/cifar10_stats.npz', 'FID cache')
# generate
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain)['net_G'])
    net_G.eval()

    counter = 0
    os.makedirs(FLAGS.output)
    with torch.no_grad():
        for start in trange(
                0, FLAGS.num_images, FLAGS.batch_size, dynamic_ncols=True):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - start)
            z = torch.randn(batch_size, FLAGS.z_dim).to(device)
            x = net_G(z).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(
                    image, os.path.join(FLAGS.output, '%d.png' % counter))
                counter += 1


def train(log_txt):
    if FLAGS.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda x: x + torch.rand_like(x) / 128)
            ]))
    if FLAGS.dataset == 'stl10':
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda x: x + torch.rand_like(x) / 128)
            ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4,
        drop_last=True)

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    loss_fn = loss_fns[FLAGS.loss]()

    optim_G = Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)
    sched_G = optim.lr_scheduler.LambdaLR(
        optim_G, lambda step: 1 - step / FLAGS.total_steps)
    sched_D = optim.lr_scheduler.LambdaLR(
        optim_D, lambda step: 1 - step / FLAGS.total_steps)

    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', grid)

    looper = infiniteloop(dataloader)

    # Filling grad buffers for the first time of using LEAD
    if FLAGS.lead_alpha_dis != 0.0:
        z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
        loss = loss_fn(net_D(net_G(z)))

        gradsG = torch.autograd.grad(
            outputs=loss, inputs=(net_G.parameters()),
            create_graph=True)
        for p, g in zip(net_G.parameters(), gradsG):
            p.grad = g * 0.0
        gen_params_flatten = parameters_to_vector(net_G.parameters())

    with trange(1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            # Discriminator
            for _ in range(FLAGS.n_dis):
                with torch.no_grad():
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    fake = net_G(z).detach()
                real = next(looper).to(device)
                net_D_real = net_D(real)
                net_D_fake = net_D(fake)
                loss = loss_fn(net_D_real, net_D_fake)

                gradsD = torch.autograd.grad(
                    outputs=loss, inputs=(net_D.parameters()),
                    create_graph=True)
                for p, g in zip(net_D.parameters(), gradsD):
                    p.grad = g

                optim_D.step()

                if FLAGS.lead_alpha_dis != 0.0:
                    gen_params_flatten_prev = gen_params_flatten + 0.0
                    gen_params_flatten = parameters_to_vector(net_G.parameters()) + 0.0
                    grad_gen_params_flatten = parameters_grad_to_vector(net_G.parameters())
                    delta_gen_params_flatten = gen_params_flatten - gen_params_flatten_prev
                    vjp_dis = torch.autograd.grad(
                        grad_gen_params_flatten, net_D.parameters(),
                        grad_outputs=delta_gen_params_flatten, create_graph=True)
                    for p, vjp in zip(net_D.parameters(), vjp_dis):
                        p.data = p.data - FLAGS.lr_D * FLAGS.lead_alpha_dis * vjp

                if FLAGS.reg != 0.0:
                    loss_G = loss_fn(net_D(net_G(z)))
                    grad_G = torch.autograd.grad(
                        loss_G, inputs=(net_G.parameters()), create_graph=True)
                    gen_grad_norm = 0.0
                    for grad in grad_G:
                        gen_grad_norm += grad.pow(2).sum()
                    grad_d_for_reg = torch.autograd.grad(
                        gen_grad_norm, inputs=(net_D.parameters()), create_graph=True)

                    for p, g in zip(net_D.parameters(), grad_d_for_reg):
                        p.data = p.data - FLAGS.lr_D * FLAGS.reg * g

                if FLAGS.loss == 'was':
                    loss = -loss
                pbar.set_postfix(loss='%.4f' % loss)
            writer.add_scalar('loss', loss, step)

            # Generator
            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
            loss = loss_fn(net_D(net_G(z)))

            gradsG = torch.autograd.grad(
                outputs=loss, inputs=(net_G.parameters()),
                create_graph=True)
            for p, g in zip(net_G.parameters(), gradsG):
                p.grad = g

            optim_G.step()

            sched_G.step()
            sched_D.step()
            pbar.update(1)

            if step == 1 or step % FLAGS.sample_step == 0:
                fake = net_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step == 1 or step % FLAGS.eval_step == 0:
                torch.save({
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                }, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.record:
                    imgs = generate_imgs(
                        net_G, device, FLAGS.z_dim, 50000, FLAGS.batch_size)
                    is_score, fid_score = get_inception_and_fid_score(
                        imgs, device, FLAGS.fid_cache, verbose=True)
                    pbar.write(
                        "%s/%s Inception Score: %.3f(%.5f), "
                        "FID Score: %6.3f" % (
                            step, FLAGS.total_steps, is_score[0], is_score[1],
                            fid_score))
                    writer.add_scalar('inception_score', is_score[0], step)
                    writer.add_scalar('inception_score_std', is_score[1], step)
                    writer.add_scalar('fid_score', fid_score, step)

                    log_txt.write('inception_score: ' + str(is_score[0]) +
                                  ' / ' + str(is_score[1]) + ' / '
                                  'fid_score: ' + str(fid_score) + '\n')
                    log_txt.flush()
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    log_txt = open('/home/pezeshki/scratch/reyhane/pytorch-gan-collections/results/' +
                   FLAGS.logdir.split('/')[-1] + ".txt", "w")
    if FLAGS.generate:
        generate()
    else:
        train(log_txt)


if __name__ == '__main__':
    app.run(main)
