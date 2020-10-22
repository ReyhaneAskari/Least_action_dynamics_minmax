import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
import sys
from utils.tensorboard_logger import Logger
from utils.inception_score import get_inception_score
from inception_score import calc_inception_score
from itertools import chain
from torchvision import utils
from torch.nn.utils import parameters_to_vector
import numpy as np
sys.path.append("..")
from utils import optim


class Generator(torch.nn.Module):
    def __init__(self, channels):
        # import ipdb; ipdb.set_trace()
        super(Generator, self).__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            # cause changed loss to BCEWithLogitsLoss from BCELoss
            # nn.Sigmoid()
            )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class DCGAN_MODEL(object):
    def __init__(self, args):
        print("DCGAN model initalization.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels
        self.mode = args.mode

        self.name = ('res/_mode_' + str(args.mode) +
                     '_beta_g_' + str(args.beta_g) +
                     '_beta_g_' + str(args.beta_g) +
                     '_beta_d_' + str(args.beta_d) +
                     '_lr_g_' + str(args.lr_g) +
                     '_lr_d_' + str(args.lr_d) +
                     '_alpha_d_vjp_' + str(args.alpha_d_vjp) +
                     '_alpha_g_vjp_' + str(args.alpha_g_vjp) +
                     '_alpha_d_grad_' + str(args.alpha_d_grad) +
                     '_alpha_g_grad_' + str(args.alpha_g_grad)
                     )
        print(self.name)
        if not os.path.exists(self.name):
            os.makedirs(self.name)
        # binary cross entropy loss and optimizer
        self.loss = nn.BCEWithLogitsLoss()

        self.cuda = "False"
        self.cuda_index = 0
        # check if cuda is available
        self.check_cuda(args.cuda)

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        if self.mode == 'adam':
            self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002,
                                                betas=(0.5, 0.999))
            self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002,
                                                betas=(0.5, 0.999))
        elif self.mode == 'adam_vjp':
            self.d_optimizer = optim.VJP_Adam(self.D.parameters(),
                                              lr=args.lr_d,
                                              betas=(args.beta_d, 0.999),
                                              alpha_vjp=args.alpha_d_vjp,
                                              alpha_grad=args.alpha_d_grad)
            self.g_optimizer = optim.VJP_Adam(self.G.parameters(),
                                              lr=args.lr_g,
                                              betas=(args.beta_g, 0.999),
                                              alpha_vjp=args.alpha_g_vjp,
                                              alpha_grad=args.alpha_g_grad)
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        # Set the logger
        self.logger = Logger('./logs')
        self.number_of_images = 10

    # cuda support
    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCEWithLogitsLoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ")
            print(self.cuda)

    def train(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0
        self.file = open("inception_score_graph.txt", "w")
        dis_params_flatten = parameters_to_vector(self.D.parameters())
        gen_params_flatten = parameters_to_vector(self.G.parameters())

        # just to fill the empty grad buffers
        if self.cuda:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        else:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1))
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        fake_labels = torch.zeros(self.batch_size)
        fake_labels = Variable(fake_labels).cuda(self.cuda_index)
        d_loss_fake = self.loss(outputs.squeeze(), fake_labels)
        (0.0 * d_loss_fake).backward(create_graph=True)
        d_loss_fake = 0.0
        best_inception_score = 0.0
        d_loss_list = []
        g_loss_list = []
        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, 100, 1, 1))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)

                if self.cuda:
                    images, z = Variable(images).cuda(self.cuda_index), Variable(z).cuda(self.cuda_index)
                    real_labels, fake_labels = Variable(real_labels).cuda(self.cuda_index), Variable(fake_labels).cuda(self.cuda_index)
                else:
                    images, z = Variable(images), Variable(z)
                    real_labels, fake_labels = Variable(real_labels), Variable(fake_labels)

                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D(images)
                d_loss_real = self.loss(outputs.squeeze(), real_labels)
                real_score = outputs

                # Compute BCE Loss using fake images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs.squeeze(), fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                if self.mode == 'adam':
                    self.D.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                elif self.mode == 'adam_vjp':
                    gradsD = torch.autograd.grad(
                        outputs=d_loss, inputs=(self.D.parameters()),
                        create_graph=True)
                    for p, g in zip(self.D.parameters(), gradsD):
                        p.grad = g
                    gen_params_flatten_prev = gen_params_flatten + 0.0
                    gen_params_flatten = parameters_to_vector(self.G.parameters()) + 0.0
                    grad_gen_params_flatten = optim.parameters_grad_to_vector(self.G.parameters())
                    delta_gen_params_flatten = gen_params_flatten - gen_params_flatten_prev
                    vjp_dis = torch.autograd.grad(
                        grad_gen_params_flatten, self.D.parameters(),
                        grad_outputs=delta_gen_params_flatten)
                    self.d_optimizer.step(vjps=vjp_dis)

                # Train generator
                # Compute loss with fake images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                # non-zero_sum
                g_loss = self.loss(outputs.squeeze(), real_labels)
                # zer_sum:
                # g_loss = - self.loss(outputs.squeeze(), fake_labels)
                # Optimize generator
                if self.mode == 'adam':
                    self.D.zero_grad()
                    self.G.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                elif self.mode == 'adam_vjp':
                    gradsG = torch.autograd.grad(
                        outputs=g_loss, inputs=(self.G.parameters()),
                        create_graph=True)
                    for p, g in zip(self.G.parameters(), gradsG):
                        p.grad = g

                    dis_params_flatten_prev = dis_params_flatten + 0.0
                    dis_params_flatten = parameters_to_vector(self.D.parameters()) + 0.0
                    grad_dis_params_flatten = optim.parameters_grad_to_vector(self.D.parameters())
                    delta_dis_params_flatten = dis_params_flatten - dis_params_flatten_prev
                    vjp_gen = torch.autograd.grad(
                        grad_dis_params_flatten, self.G.parameters(),
                        grad_outputs=delta_dis_params_flatten)
                    self.g_optimizer.step(vjps=vjp_gen)

                generator_iter += 1

                if generator_iter % 1000 == 0:
                    # Workaround because graphic card memory can't store more than 800+ examples in memory for generating image
                    # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                    # This way Inception score is more correct since there are different generated examples from every class of Inception model
                    sample_list = []
                    for i in range(10):
                        z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                        samples = self.G(z)
                        sample_list.append(samples.data.cpu().numpy())

                    # Flattening list of lists into one list of numpy arrays
                    new_sample_list = list(chain.from_iterable(sample_list))
                    print("Calculating Inception Score over 8k generated images")
                    # Feeding list of numpy arrays
                    inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                                                          resize=True, splits=10)
                    print('Epoch-{}'.format(epoch + 1))
                    print(inception_score)
                    if inception_score >= best_inception_score:
                        best_inception_score = inception_score
                        self.save_model()

                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:64]
                    grid = utils.make_grid(samples)
                    utils.save_image(grid, self.name + '/iter_{}_inception_{}_.png'.format(str(generator_iter).zfill(3), str(inception_score)))

                    time = t.time() - self.t_begin
                    print("Inception score: {}".format(inception_score))
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(time))

                    # Write to file inception_score, gen_iters, time
                    output = str(generator_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                    self.file.write(output)

                if ((i + 1) % 100) == 0:
                    d_loss_list += [d_loss.item()]
                    g_loss_list += [g_loss.item()]
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.item(), g_loss.item()))

                    z = Variable(torch.randn(self.batch_size, 100, 1, 1).cuda(self.cuda_index))

                    # TensorBoard logging
                    # Log the scalar values
                    info = {
                        'd_loss': d_loss.item(),
                        'g_loss': g_loss.item()
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, generator_iter)

                    # Log values and gradients of the parameters
                    for tag, value in self.D.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, self.to_np(value), generator_iter)
                        self.logger.histo_summary(tag + '/grad', self.to_np(value.grad), generator_iter)

                    # Log the images while training
                    info = {
                        'real_images': self.real_images(images, self.number_of_images),
                        'generated_images': self.generate_img(z, self.number_of_images)
                    }

                    for tag, images in info.items():
                        self.logger.image_summary(tag, images, generator_iter)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_final_model()
        np.save(self.name + '/d_loss', np.array(d_loss_list))
        np.save(self.name + '/g_loss', np.array(g_loss_list))
        self.evaluate(
            train_loader, self.name + '/discriminator.pkl',
            self.name + '/generator.pkl')

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        self.G.eval()
        all_fake = []
        for i in range(10):
            z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
            fake = self.G(z)
            all_fake += [fake]
        all_fake = torch.cat(all_fake, 0)
        inception_score = calc_inception_score((all_fake.cpu().data.numpy() + 1.0) * 128)
        print(inception_score)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, self.name + '/best_inception_score' + str(inception_score) + '.png')

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), self.name + '/generator.pkl')
        torch.save(self.D.state_dict(), self.name + '/discriminator.pkl')
        print('Models save to generator.pkl & discriminator.pkl ')

    def save_final_model(self):
            torch.save(self.G.state_dict(), self.name + '/final_generator.pkl')
            torch.save(self.D.state_dict(), self.name + '/final_discriminator.pkl')
            print('Final models save to generator.pkl & discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1 * alpha + z2 * (1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.C, 32, 32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))
