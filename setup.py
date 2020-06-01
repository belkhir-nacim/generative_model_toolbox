from setuptools import setup

setup(
    name='generative_model_toolbox',
    version='0.1',
    packages=['generative_models_toolbox',
              'generative_models_toolbox.algos', 'generative_models_toolbox.algos.ae', 'generative_models_toolbox.algos.gan',
              'generative_models_toolbox.algos.vae', 'generative_models_toolbox.algos.vqvae', 'generative_models_toolbox.algos.pixelcnn',
              'generative_models_toolbox.algos.graphicalmodel', 'generative_models_toolbox.utils', 'generative_models_toolbox.layers',
              'generative_models_toolbox.vqvae2', 'generative_models_toolbox.sampler', 'generative_models_toolbox.trainer',
              'generative_models_toolbox.metric_loss'],
    url='https://github.com/belkhir-nacim/generative_model_toolbox',
    license='CECILL-2.1',
    author='Nacim Belkhir',
    author_email='belkhir.nacim@gmail.com',
    description='a set of implementations related to generative models in deep learning, including auto encoders, variational auto encoders, gans, and auto regressive models'
)
