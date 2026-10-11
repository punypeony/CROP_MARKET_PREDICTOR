
import joblib

MODEL_PATH = "models/best_model.joblib"

def main():
    model = joblib.load(MODEL_PATH)

    print("Model type:", type(model).__name__)
    print("Pipeline steps:", model.named_steps.keys())

    preprocessor = model.named_steps["preprocessor"]
    print("\nExpected input columns:")
    print(list(preprocessor.feature_names_in_))

    print("\nCategorical categories learned by the model:")
    encoder = preprocessor.named_transformers_["categorical"]

    for column, categories in zip(
        preprocessor.transformers_[0][2],
        encoder.categories_,
    ):
        print(f"\n{column} ({len(categories)} categories)")
        print(list(categories[:15]))

    print("\nModel inspection complete. No training or saving performed.")

if __name__ == "__main__":
    main()
