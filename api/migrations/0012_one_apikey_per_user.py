# Hand-written migration — change ApiKey.user from ForeignKey to OneToOneField.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0011_add_api_key_model'),
    ]

    operations = [
        migrations.AlterField(
            model_name='apikey',
            name='user',
            field=models.OneToOneField(
                help_text='The API user (IP address) this key belongs to.',
                on_delete=django.db.models.deletion.CASCADE,
                related_name='api_key',
                to='api.apiuser',
            ),
        ),
    ]
