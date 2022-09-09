<script setup lang="ts">
</script>

<template>
  <n-grid :cols="2">
    <n-gi>
      <n-space vertical align="center"
        ><n-h1 style="color: white; font-weight: 400">Christopher Smith</n-h1>
        <n-avatar round src="src/assets/ProfessionalPic.jpg" :size="400"
      /></n-space>
    </n-gi>
    <n-gi>
      <n-space vertical align="center" style="width: 600px"></n-space>
      <n-h2 style="color: white"
        >I am an undergradute student studying Computer Science at The
        University of Texas at Arlington.</n-h2
      >
      <n-h2 style="color: white"
        >I am currently employed at Ayoka Systems as an intern for Web
        Application Development.</n-h2
      >
      <n-h2 style="color: white">
        I am currently interested in AI and Web Application Development.
      </n-h2>
    </n-gi>
  </n-grid>
</template>
