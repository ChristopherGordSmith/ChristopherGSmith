<script setup lang="ts">
import KaggleTutorial from './blogs/KaggleTutorial.vue';
import ImageClassifier from './blogs/ImageClassifier.vue'
import NaiveBayesClassifier from './blogs/NaiveBayesClassifier.vue';
</script>

<template>
  <n-space style="padding: 50px 100px 50px 100px">
    <n-collapse default-expanded-names="1" accordion>
      <n-collapse-item >
        <template #header>
          <n-h1 style="color: #e4572e; font-size: 40px;">Kaggle Titanic Tutorial</n-h1>
        </template>
        <KaggleTutorial />
      </n-collapse-item>
      <n-collapse-item>
        <template #header>
          <n-h1 style="color: #e4572e; font-size: 40px;">Image Classifier</n-h1>
        </template>
        <ImageClassifier />
      </n-collapse-item>
      <n-collapse-item>
        <template #header>
          <n-h1 style="color: #e4572e; font-size: 40px;">Naive Bayes Classifier</n-h1>
        </template>
        <NaiveBayesClassifier />
      </n-collapse-item>
    </n-collapse>
  </n-space>
</template>

<style>
.n-base-icon {
  padding-top: 10px;
  color: white
}
</style>