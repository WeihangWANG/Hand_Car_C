/*****************************************************************//**
 *       @file  sample_capture_frames.c
 *      @brief  Brief Decsription
 *
 *  Detail Decsription starts here
 *
 *   @internal
 *     Project  $Project$
 *     Created  3/24/2017 
 *    Revision  $Id$
 *     Company  Data Miracle, Shanghai
 *   Copyright  (C) 2017 Data Miracle Intelligent Technologies
 *    
 *    THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 *    KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *    IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 *    PARTICULAR PURPOSE.
 *
 * *******************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/types.h>

#include <stdbool.h>
#include <assert.h>
#include "unistd.h"
#include "getopt.h"
#include "pthread.h"
#include "dmcam.h"

#ifdef _WIN32
    #include <windows.h>
#define sleep(s) Sleep(s*1000)
#else
#endif

#define FRAME_SIZE (320*240*4 * 2)
#define FRAME_BUF_FCNT 16

static dmcam_dev_t *dev;
#ifdef _MSC_VER
    #pragma comment(lib, "pthreadVC2.lib")
#endif
static void on_frame_rdy(dmcam_dev_t *dev, dmcam_frame_t *f)
{
    printf("cap: idx=%d, size=%d\n", f->frame_info.frame_idx, f->frame_info.frame_size);
    //usleep(300000);
}

/* return false will stop the capturing process */
static bool on_cap_err(dmcam_dev_t *dev, int err, void *err_arg)
{
    printf("cap error found : %s, arg=%p\n", dmcam_error_name(err), err_arg);
    switch (err) {
        case DMCAM_ERR_CAP_FRAME_DISCARD:
            printf("  data process too slow: total missing %d frames\n", (int)(size_t)err_arg);
            break;
        case DMCAM_ERR_CAP_STALL:
            printf(" usb pipe stall!\n");
            break;
        default:
            break;
    }
    return true;
}
static pthread_t test_th;
static void* th_test_entry(void *arg)
{
    uint32_t after_msec = (size_t)arg;

    printf(" stop after %d sec\n", after_msec);
    sleep(after_msec);

    dmcam_cap_stop(dev);

    printf(" exit async stop\n");
    return NULL;
}

static void async_stop(uint32_t after_msec)
{
    if (pthread_create(&test_th, NULL, th_test_entry, (void *)(size_t)after_msec) < 0) {
        printf("create test thread failed\n");
    }
}

static void async_wait(void)
{
    pthread_join(test_th, NULL);
}

void test_async_stop(void)
{
    int i;

    printf("---- test async stop ---\n");
    for (i = 0; i < 1; i++) {
        int n;

        dmcam_cap_start(dev);
        async_stop(2);
        for (n = 0; n < 10; n++) {
            int fr_cnt = dmcam_cap_get_frames(dev, 45, NULL, 0, NULL);

            printf("get %d frames\n", fr_cnt);
            if (fr_cnt < 45) {
                printf("get frame stopped!\n");
                break;
            }
        }
        dmcam_cap_wait(dev, 10000);
        async_wait();
        sleep(1);
    }
}
/**
 * this programming model is deprecated.
 * 
 */

void test_model_deprecated(void)
{
    dmcam_cap_set_callback_on_frame_ready(dev, on_frame_rdy);

    {
        printf("---- test paradigm: start/wait/stop  ---\n");
        fprintf(stderr, " start...\n");
        assert(dmcam_cap_start(dev));
        dmcam_cap_wait(dev, 10000);
        //dmcam_cap_wait(dev, 0);
        fprintf(stderr, "stop...\n");
        assert(dmcam_cap_stop(dev));
    }
}

/**
 * this is the recommended programming model. with full 
 * parameter usage. 
 */
void test_model_new(uint32_t frame_num)
{
    dmcam_frame_t fbuf_info;
    uint8_t *fbuf = malloc(FRAME_SIZE * frame_num);
    int total_fr = 0;
    int fr_cnt;
    uint32_t offset = 0;
    assert(fbuf);
    printf("---- test paradigm: start/get_frames/stop  ---\n");
    dmcam_cap_set_callback_on_frame_ready(dev, NULL); // optional: disable frame ready callback
    dmcam_cap_start(dev);

    /* get frame_num frames */
    fr_cnt = dmcam_cap_get_frames(dev, frame_num, fbuf, FRAME_SIZE * frame_num, &fbuf_info);
    total_fr += fr_cnt;
    printf("get %d frames: [%u, %ux%u, %u]\n",
           fr_cnt, fbuf_info.frame_info.frame_idx,
           fbuf_info.frame_info.width, fbuf_info.frame_info.height, fbuf_info.frame_info.frame_format);
    if (fr_cnt < frame_num) { // less frames means sampling is stopped.
        printf("capturing is stopped due to some error!\n");
    }
    if (total_fr > frame_num) {
        printf("get enough frames, we stop\n");
    }
    // do some work to process every frame in fbuf
    printf("proc frames .... frame size %d\n", fbuf_info.frame_info.frame_size);

    /* decode one frame to distance */
    {
        int i;
        int dist_len = fbuf_info.frame_info.width * fbuf_info.frame_info.height;
        float *dist = malloc(sizeof(float) * dist_len);
        uint16_t *idist = malloc(sizeof(uint16_t) * dist_len);
        FILE *fid = fopen("frames.bin", "wb+");
        if (fid == NULL) {
            return;
        }
        for (i = 0; i < fr_cnt; i++) {
            {
                int calc_len = dmcam_frame_get_distance(dev, dist, dist_len, fbuf + offset, fbuf_info.frame_info.frame_size, &fbuf_info.frame_info);
                assert(calc_len == dist_len);
                for (int j = 0; j < dist_len; j++) {
                    idist[j] = (uint16_t)(dist[j] * 1000);
                }
				
                fwrite(idist, 2, dist_len, fid);
                offset += FRAME_SIZE;
            }
        }
        fclose(fid);
        free(dist);
        free(idist);
    }
    free(fbuf);
    dmcam_cap_stop(dev);
}

void test_model_new_snapshot(void)
{
    uint8_t *fbuf = malloc(FRAME_SIZE);

    assert(fbuf);
    printf("---- test paradigm: snapshot  ---\n");
    dmcam_cap_set_callback_on_frame_ready(dev, NULL); // optional: disable frame ready callback
    printf(" * snapshot in stop status \n");
    dmcam_cap_stop(dev);
    if (!dmcam_cap_snapshot(dev, fbuf, FRAME_SIZE, NULL)) {
        printf("snapshot failed!\n");
    }
    printf("get frame @ %p\n", fbuf);

    //printf(" * snapshot in start status \n");
    {
        //dmcam_frame_t fbuf_info;
        //dmcam_cap_start(dev);
        //if (!dmcam_cap_snapshot(dev, fbuf, FRAME_SIZE, &fbuf_info)) {
        //    printf("snapshot failed!\n");
        //}
        //printf("get frame @ %p, fr_cnt=%u\n", fbuf, fbuf_info.frame_count);
        //usleep(300 * 1000);
        //if (!dmcam_cap_snapshot(dev, fbuf, FRAME_SIZE, &fbuf_info)) {
        //    printf("snapshot failed!\n");
        //}
        //printf("get frame @ %p, fr_cnt=%u\n", fbuf, fbuf_info.frame_count);
        //dmcam_cap_stop(dev);
    }
    free(fbuf);
}

int main(int argc, char **argv)
{
    int debug_level = 0;
    uint32_t frm_num = 100, intg_tim = 100;
    if (argc > 1) {
        debug_level = atoi(argv[1]);
        printf(" debug level set to %d\n", debug_level);
    }

    frm_num = atoi(argv[2]);
    frm_num =  frm_num > 400 ? 400 : frm_num;
    printf("Get %d frames\n", frm_num);
    intg_tim = atoi(argv[3]);
    intg_tim = intg_tim > 1500 ? 1500 : intg_tim;
    printf("Set intg time %d us\n", intg_tim);
    dmcam_init(NULL);

    dmcam_log_cfg(LOG_LEVEL_INFO, LOG_LEVEL_TRACE, LOG_LEVEL_NONE - debug_level);

#if 1
    {
        int dev_cnt;
        dmcam_dev_t dev_list[4];

        dev_cnt = dmcam_dev_list(dev_list, 4);

        printf(" %d dmcam device found\n", dev_cnt);
        if (dev_cnt == 0)
            goto FINAL;
        /* open device  */
        dev = dmcam_dev_open(&dev_list[0]);
        if (!dev) {
            printf(" open device failed\n");
            goto FINAL;
        }
        /* reset device */

        //dmcam_dev_reset(dev, DEV_RST_TFC);
        /* close device */
        dmcam_dev_close(dev);
    }
#endif
    /* open device */
    dev = dmcam_dev_open(NULL);
    if (!dev) {
        printf(" open device failed\n");
        goto FINAL;
    }
    /* set illumination power*/
    {
        dmcam_param_item_t wparam;

        memset(&wparam, 0, sizeof(wparam));
        wparam.param_id = PARAM_INTG_TIME;
        wparam.param_val_len = 1;
        wparam.param_val.intg.intg_us = intg_tim;
        assert(dmcam_param_batch_set(dev, &wparam, 1));
    }
    /* capture frames using interval alloced buffer */
    dmcam_cap_set_frame_buffer(dev, NULL, FRAME_SIZE * FRAME_BUF_FCNT);
    /* set error callback for capturing */
    dmcam_cap_set_callback_on_error(dev, on_cap_err);

    /* reset the usbif will help to fix stall problem */
    //dmcam_dev_reset(dev, DEV_RST_USB);

    //  test_model_deprecated();
    test_model_new(frm_num);
    //test_model_new_snapshot();
    //
    //test_async_stop();

    dmcam_dev_close(dev);
FINAL:
    dmcam_uninit();
    return 0;
}
