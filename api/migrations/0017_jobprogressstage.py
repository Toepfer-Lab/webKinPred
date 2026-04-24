from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0016_job_start_time"),
    ]

    operations = [
        migrations.CreateModel(
            name="JobProgressStage",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("stage_index", models.PositiveIntegerField()),
                ("target", models.CharField(max_length=32)),
                ("method_key", models.CharField(max_length=50)),
                ("method_display_name", models.CharField(blank=True, default="", max_length=100)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "pending"),
                            ("running", "running"),
                            ("completed", "completed"),
                            ("failed", "failed"),
                            ("skipped", "skipped"),
                        ],
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("message", models.TextField(blank=True, default="")),
                ("molecules_total", models.IntegerField(default=0)),
                ("molecules_processed", models.IntegerField(default=0)),
                ("invalid_rows", models.IntegerField(default=0)),
                ("predictions_total", models.IntegerField(default=0)),
                ("predictions_made", models.IntegerField(default=0)),
                ("embedding_enabled", models.BooleanField(default=False)),
                (
                    "embedding_state",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("", ""),
                            ("not_required", "not_required"),
                            ("pending", "pending"),
                            ("running", "running"),
                            ("done", "done"),
                            ("error", "error"),
                        ],
                        default="",
                        max_length=20,
                    ),
                ),
                ("embedding_method_key", models.CharField(blank=True, default="", max_length=50)),
                ("embedding_target", models.CharField(blank=True, default="", max_length=32)),
                ("embedding_total", models.IntegerField(default=0)),
                ("embedding_cached_already", models.IntegerField(default=0)),
                ("embedding_need_computation", models.IntegerField(default=0)),
                ("embedding_computed", models.IntegerField(default=0)),
                ("embedding_remaining", models.IntegerField(default=0)),
                (
                    "job",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="progress_stages",
                        to="api.job",
                    ),
                ),
            ],
            options={
                "ordering": ["job_id", "stage_index"],
            },
        ),
        migrations.AddConstraint(
            model_name="jobprogressstage",
            constraint=models.UniqueConstraint(
                fields=("job", "stage_index"),
                name="api_jobprogressstage_unique_job_stage_index",
            ),
        ),
        migrations.AddConstraint(
            model_name="jobprogressstage",
            constraint=models.UniqueConstraint(
                fields=("job", "target"),
                name="api_jobprogressstage_unique_job_target",
            ),
        ),
    ]
