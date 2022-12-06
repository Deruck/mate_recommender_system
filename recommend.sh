python ./src/recommender.py \
    --model fork \
    --recommend_users ./data/pre_processed/survey_new.csv \
    --user_pool ./data/pre_processed/survey_new.csv \
    --output_file ./out/recommendations.json \
    --top 10