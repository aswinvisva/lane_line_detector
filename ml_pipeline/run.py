from ml_pipeline.feature_generator import Generator
from ml_pipeline.models import Models

if __name__ == '__main__':

    # model = Models(model="BCDU_net_D3")
    # model.fit_model(shuffle=False)

    gen = Generator()
    results = gen.evaluate_generator(model="BCDU_net_D3", option="jiqing")
    print(results)
