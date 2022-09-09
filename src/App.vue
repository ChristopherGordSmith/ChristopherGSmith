<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
</script>

<template>
  <n-layout position="absolute" :native-scrollbar="false">
    <n-layout-header mode="horizontal" style="background: #e4572e">
      <n-grid style="height: 50px; width: 100%" :cols="12">
        <n-gi :span="1">
          <n-space justify="space-around" align="center" style="height: 100%">
            <router-link class="topNavigation" to="/">Home</router-link>
            <router-link class="topNavigation" to="/about">Blogs</router-link>
          </n-space>
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
        :cols="2"
        item-responsive
        :x-gap="100"
      >
        <n-gi>
          <n-space vertical align="center">
            <n-h3>Address: Mansfield, TX, 76063</n-h3>
            <n-h3>Phone: (682)-558-3975</n-h3>
          </n-space>
        </n-gi>
        <n-gi>
          <n-space vertical align="center">
            <n-h3>Email: christophergsmith3975@gmail.com</n-h3>
            <n-h3
              >LinkedIn Profile:
              <a
                href="https://www.linkedin.com/in/christophergsmith3975"
                target="_blank"
                >www.linkedin.com/in/christophergsmith3975</a
              ></n-h3
            >
          </n-space>
        </n-gi>
      </n-grid>
    </n-layout-footer>
  </n-layout>
</template>

<style scoped>
</style>
