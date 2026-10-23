from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("recognition", "0003_dataset_task_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="trainingrun",
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
    ]
