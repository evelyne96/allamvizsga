# Generated by Django 2.0.1 on 2018-03-08 13:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authenticate', '0004_profile_mood'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='companion',
            field=models.CharField(default='images/female/Koko/talk.png', max_length=254),
        ),
    ]
