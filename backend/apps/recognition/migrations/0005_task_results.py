from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("recognition", "0004_trainingrun_task_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="trainedmodel",
            name="task_type",
            field=models.CharField(
                choices=[
                    ("classification", "图片识别"),
                    ("detection", "目标检测"),
                    ("segmentation", "语义分割"),
                ],
                default="classification",
                max_length=20,
                verbose_name="功能",
            ),
        ),
        migrations.AddField(
            model_name="predictionrecord",
            name="task_type",
            field=models.CharField(
                choices=[
                    ("classification", "图片识别"),
                    ("detection", "目标检测"),
                    ("segmentation", "语义分割"),
                ],
                default="classification",
                max_length=20,
                verbose_name="功能",
            ),
        ),
        migrations.AddField(
            model_name="predictionrecord",
            name="result_data",
            field=models.JSONField(blank=True, default=dict, verbose_name="结构化结果"),
        ),
    ]
