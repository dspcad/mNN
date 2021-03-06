import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import java.util.Random;

import Jama.Matrix;

public class TwoLayerNN {
    public static final int NUM_SAMPLE = 10000;

    public static void main(String[] args) throws IOException {
        int [] train_label_1 = new int[10000];
        //int [] train_label_2 = new int[10000];
        //int [] train_label_3 = new int[10000];
        //int [] train_label_4 = new int[10000];
        //int [] train_label_5 = new int[10000];

        FileInputStream inputStream = new FileInputStream("../../cifar-10-batches-bin/data_batch_1.bin");
        ArrayList<int[]> train_images_1 = batchRead(inputStream, train_label_1);

        //inputStream = new FileInputStream("../../cifar-10-batches-bin/data_batch_2.bin");
        //ArrayList<int[]> train_images_2 = batchRead(inputStream, train_label_2);

        //inputStream = new FileInputStream("../../cifar-10-batches-bin/data_batch_3.bin");
        //ArrayList<int[]> train_images_3 = batchRead(inputStream, train_label_3);

        //inputStream = new FileInputStream("../../cifar-10-batches-bin/data_batch_4.bin");
        //ArrayList<int[]> train_images_4 = batchRead(inputStream, train_label_4);

        //inputStream = new FileInputStream("../../cifar-10-batches-bin/data_batch_5.bin");
        //ArrayList<int[]> train_images_5 = batchRead(inputStream, train_label_5);

        Matrix train_data_matrix = new Matrix(3072,10000);
        fillValuesToMatrix(train_data_matrix, train_images_1, 0);


        Matrix weight_matrix_1   = new Matrix(50,3072);
        Matrix weight_matrix_2   = new Matrix(10,50);
        double [] b_1            = new double[50];
        double [] b_2            = new double[10];

        initializeWeights(weight_matrix_1, 50, 3072);
        initializeWeights(weight_matrix_2, 10,  50);
        initializeBias(b_1);
        initializeBias(b_2);
        
        double step_size    = 0.0005;
        double reg_strength = 0.001;

        double [][] exp_scores;
        double data_loss;
        double reg_loss;
        double loss;

        double [] db_1;
        double [] db_2;

        Matrix hidden_layer;
        Matrix dscores;
        Matrix dW2;
        Matrix dW1;
        Matrix dhidden;
         
 
        for(int i=0; i<100; i++){
            hidden_layer = ReLU(weight_matrix_1, train_data_matrix, b_1); //50 x 10000
            exp_scores = scoreFun(weight_matrix_2, hidden_layer, b_2);
            interpretInProbs(exp_scores);
            data_loss = lossFun(exp_scores, train_label_1);
            reg_loss = 0.5*reg_strength*regLoss(weight_matrix_2);
            loss = data_loss + reg_loss;

            if(i % 10 == 0)
                System.out.println("iteration "+i+": loss: "+loss+" training accuracy: "+evalAccuracy(exp_scores, train_label_1));
            
            
        
            dscores = gradientComputation(exp_scores, train_label_1); //10 x 10000
            dW2 = dscores.times(hidden_layer.transpose()); //10 x 50
            db_2 = gradientBias(dscores, 10);

            //if(i % 10 == 0)
            //System.out.println("hidden layer "+calNonzero(hidden_layer));

            dhidden = weight_matrix_2.transpose().times(dscores); //50 x 10000
            backPropgReLU(dhidden, hidden_layer);

            dW1 = dhidden.times(train_data_matrix.transpose());
            db_1 = gradientBias(dhidden, 50);
            
            addRegGradientContrbution(dW2, weight_matrix_2, reg_strength);
            addRegGradientContrbution(dW1, weight_matrix_1, reg_strength);
     
            updateWights(dW2, weight_matrix_2, step_size);
            updateWights(dW1, weight_matrix_1, step_size);

            updateBias(db_2, b_2, step_size);
            updateBias(db_1, b_1, step_size);
            
        }
        
    }

    public static int calNonzero(Matrix M){
        int row = M.getRowDimension();
        int col = M.getColumnDimension();
        int sum = 0;

        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++)
                if(M.get(i,j)>0)
                    sum++;

        return sum;
          
    }

    public static void backPropgReLU(Matrix dH, Matrix H){
        int row = dH.getRowDimension();
        int col = dH.getColumnDimension();

        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++)
                if(H.get(i,j) <= 0){
                    dH.set(i,j, 0.0);
                    //System.out.println("dW: ");
                }
    }

    public static void updateBias(double [] db, double [] b, double step_size){
        for(int i=0; i< db.length; i++)
            b[i] = b[i] - step_size*db[i];
    }

    public static void updateWights(Matrix dW, Matrix weight_matrix, double step_size){
        int row = dW.getRowDimension();
        int col = dW.getColumnDimension();

        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++)
                weight_matrix.set(i,j, weight_matrix.get(i,j)-step_size*dW.get(i,j));

    }


    public static void addRegGradientContrbution(Matrix dW, Matrix weight_matrix, double reg_strength){
        int row = dW.getRowDimension();
        int col = dW.getColumnDimension();


        //System.out.println("dW: ");
        //System.out.println("row dim: "+dW.getRowDimension());
        //System.out.println("col dim: "+dW.getColumnDimension());

        //System.out.println("weight_matrix: ");
        //System.out.println("row dim: "+weight_matrix.getRowDimension());
        //System.out.println("col dim: "+weight_matrix.getColumnDimension());


        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++)
                dW.set(i,j, dW.get(i,j)+reg_strength*weight_matrix.get(i,j));
    }

    public static double [] gradientBias(Matrix M, int length){
        double [] db = new double[length];
        double sum;

        for(int i=0;i<length;i++){
            sum = 0;
            for(int j=0;j<NUM_SAMPLE;j++)
                sum += M.get(i,j);

            db[i] = sum;
        }

        return db;
    }

    public static Matrix gradientComputation(double [][] exp_scores, int [] train_label){
        Matrix dscores = new Matrix(10, NUM_SAMPLE);

        for(int j=0; j<NUM_SAMPLE;j++){
            for(int i=0;i<10;i++){
                if(i == train_label[j]){
                    dscores.set(i, j, (exp_scores[i][j] - 1)/NUM_SAMPLE);
                    //System.out.println("prob: "+dscores.get(i, j));
                }
                else 
                    dscores.set(i, j, exp_scores[i][j]/NUM_SAMPLE);
            }
        }


        return dscores;
    }


    public static double evalAccuracy(double [][] exp_scores, int [] train_label){
        int [] predict = new int[train_label.length];
        double max = 0;
        double correct_sum = 0;

        for(int j=0; j<train_label.length;j++){
            for(int i=0;i<10;i++){
                if(exp_scores[i][j] >= max){
                    max = exp_scores[i][j];
                    predict[j] = i;
                }
            }

            max = 0;

        }

        for(int j=0; j<train_label.length;j++)
            if(predict[j] == train_label[j])
                correct_sum++;
        

        return correct_sum/train_label.length;
    }


    public static double regLoss(Matrix W){
        int row = W.getRowDimension();
        int col = W.getColumnDimension();
        double sum = 0;

        for(int i=0;i<row;i++)
            for(int j=0;j<col;j++)
                sum += Math.pow(W.get(i,j), 2);

        return sum;
    }


    public static double lossFun(double [][] exp_scores, int [] train_label){
        double sum = 0;

        for(int i=0;i<NUM_SAMPLE;i++)
            sum -= Math.log(exp_scores[ train_label[i] ][i]);
        
        return sum/NUM_SAMPLE;
    }


    public static void interpretInProbs(double [][] exp_scores){
        double sum;

        for(int j=0;j<NUM_SAMPLE;j++){
            sum = 0;
            for(int i=0;i<10;i++)
                sum += exp_scores[i][j];

           
            for(int i=0;i<10;i++){
                exp_scores[i][j] = exp_scores[i][j]/sum;
                //System.out.println("prob: "+exp_scores[i][j]);
            }
        }

    }

    public static double [][] scoreFun(Matrix W, Matrix H, double [] b){
        Matrix result = W.times(H);
        double [][] scores = new double[b.length][result.getColumnDimension()];

        for(int i=0;i<b.length;i++)
            for(int j=0;j<result.getColumnDimension();j++){
                scores[i][j] =Math.exp(result.get(i,j) + b[i]);
                //scores[i][j] = result.get(i,j) + b[i];
                //System.out.println("score: "+scores[i][j]);

            }

        return scores;            
    }

    public static Matrix ReLU(Matrix W, Matrix X, double [] b){
        Matrix result = W.times(X);
        int row = result.getRowDimension();
        int col = result.getColumnDimension();
        //System.out.println("row: "+row);
        //System.out.println("col: "+col);
        double max_val = 0;

        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++){
                max_val = result.get(i,j)+b[i];
                if(max_val > 0){
                    result.set(i,j, max_val);
                    //System.out.println("max: "+max_val);
                }
                else
                    result.set(i,j, 0.0);

            }
        

        //System.out.println("row dim: "+dot_product.getRowDimension());
        //System.out.println("col dim: "+dot_product.getColumnDimension());
        return result;
    }


    public static void initializeBias(double [] b){
        for(int i=0; i<b.length; i++)
            b[i] = 0.0; 
    }


    public static void initializeWeights(Matrix W, int row, int col){
        Random normal_var = new Random();

        for(int i=0; i<row; i++)
            for(int j=0; j<col; j++)
                W.set(i, j, 0.01*normal_var.nextGaussian()/Math.sqrt(NUM_SAMPLE)); 

        //System.out.println("W[0][0]: "+W.get(0,0));
    }


    public static void fillValuesToMatrix(Matrix train_data_matrix, ArrayList<int[]> train_images, int ref_pt){
        int [] image_pixel;

        for(int j=0;j<NUM_SAMPLE;j++){
            image_pixel = train_images.get(j);
            for(int i=0;i<3072;i++)
                train_data_matrix.set(i, ref_pt*NUM_SAMPLE+j, image_pixel[i]);
        }

        //System.out.println("pixel(3071): "+train_data_matrix.get(3071,9999));
    }

    public static ArrayList<int[]> batchRead(FileInputStream inputStream, int [] train_labels) throws IOException{
        byte[] b = new byte[3073];
        inputStream.read(b);

        ArrayList<int[]> train_images = new ArrayList<int[]>();

        int i = 0;
        do {
            int [] pixels = new int[3072];
            train_labels[i] = b[0];
            //System.out.println("label: "+train_labels[i]);
            for (int row = 0; row < 32; row++) {
                for (int col = 0; col < 32; col++) {
                    Color color = new Color(
                            b[1 + 1024 * 0 + row * 32 + col] & 0xFF,
                            b[1 + 1024 * 1 + row * 32 + col] & 0xFF,
                            b[1 + 1024 * 2 + row * 32 + col] & 0xFF);
                  
                    //pixels[row*32+col*3+0] = color.getRed();  
                    //pixels[row*32+col*3+1] = color.getGreen();  
                    //pixels[row*32+col*3+2] = color.getBlue();  

                    pixels[row*32+col+0*32*32] = color.getRed();  
                    pixels[row*32+col+1*32*32] = color.getGreen();  
                    pixels[row*32+col+2*32*32] = color.getBlue();  

                   
                }
            }
            //System.out.println("pixel(3071): "+pixels[3071]);
            train_images.add(pixels);
            i++;
        } while(inputStream.read(b) != -1);
            
        System.out.println("size is "+train_images.size());

        return train_images;
    }
}
