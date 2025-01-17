from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

def build_ensemble_model(nn_preds, xgb_preds, y_test, le):
    # Create a voting classifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('nn', nn_preds),
            ('xgb', xgb_preds)
        ],
        voting='soft'  # Use 'soft' for probability-based voting
    )

    # Evaluate the ensemble model
    ensemble_preds = ensemble_model.predict(X_test_combined)
    print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_preds))
    print(classification_report(y_test, ensemble_preds, target_names=le.classes_))

    return ensemble_model