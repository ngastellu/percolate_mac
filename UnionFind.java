public class UnionFind {
    public static int[] equiv;
    public static int[] height;

    public static void init(int len){
        equiv = new int[len];
        height = new int[len];
        for (int i = 0; i < equiv.length; i++) {
            equiv[i] = i;
        }
    }

    public static int naiveFind(int x) {
        return equiv[x];
    }

    public static int fastFind(int x) {
        if (equiv[x] == x) return x;
        else return fastFind(equiv[x]);
    }

    public static int logFind(int x) {
        if (equiv[x] == x) return x;
        equiv[x] = equiv[equiv[x]];
        return logFind(equiv[x]);
    }

    public static int naiveUnion(int x, int y) {
        int x_class = naiveFind(x);
        int y_class = naiveFind(y);
        for (int i = 0; i < equiv.length; i++) {
            if (equiv[i] == x_class) {
                equiv[i] = y_class;
            }
        }
        return y_class;
    }

    public static int fastUnion(int x, int y) {
        int y_class = fastFind(y);
        int x_class = fastFind(x);
        equiv[x_class] = y_class;
        return y_class;
    }

    public static int find(int x) {
        return logFind(x);
    }

    public static int union(int x, int y) {
        return logUnion(x, y);
    }


    public static int logUnion(int x, int y) {
        int x_class = fastFind(x);
        int y_class = fastFind(y);
        if (height[x_class] > height[y_class]) {
            equiv[y_class] = x_class;
            return x_class;
        }
        equiv[x_class] = y_class;
        height[y_class] = Math.max(height[y_class], height[x_class]+1);
        return y_class;
    }

}
