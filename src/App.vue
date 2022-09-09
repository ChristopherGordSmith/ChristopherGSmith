<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
</script>

<template>
  <n-layout position="absolute" :native-scrollbar="false">
    <n-layout-header mode="horizontal" style="background: #e4572e">
      <n-grid style="height: 50px; width: 100%" :cols="4">
        <n-gi>
          <n-avatar
            src="src/assets/ResumeLogo.png"
            :size="40"
            style="background: #ffffff"
          />
        </n-gi>
        <n-gi :span="2.5" :offset="0.5">
          <router-link class="topNavigation" to="/">Home</router-link>
          <router-link class="topNavigation" to="/about">About me</router-link>
        </n-gi>
      </n-grid>
    </n-layout-header>
    <n-layout>
      <n-space
        justify="center"
        align="center"
        style="background: #011627; min-height: 750px"
      >
        <RouterView />
      </n-space>
    </n-layout>
    <n-layout-footer>
      <n-grid
        style="background: #e07a5f; min-height: 100px; width: 100%"
        :cols="3"
        item-responsive
        :x-gap="100"
      >
        <n-gi>
          <n-space vertical>
            <n-h3>item 1 </n-h3>
            <n-h3>item 2</n-h3>
          </n-space>
        </n-gi>
        <n-gi>
          <n-space vertical>
            <n-h3>item 1 </n-h3>
            <n-h3>item 2</n-h3>
          </n-space>
        </n-gi>
        <n-gi>
          <n-space vertical>
            <n-h3>item 1 </n-h3>
            <n-h3>item 2</n-h3>
          </n-space>
        </n-gi>
      </n-grid>
    </n-layout-footer>
  </n-layout>
</template>

<style scoped>
</style>
